#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

#include "Common.h"
#include "Common.cuh"
#include "Rasterization.h"

// ROCm only. float4 LDS batch (one ds_load_b128 + one ds_load_b96 off a single
// base, replacing four loads off three addresses) plus a one-ahead prefetch.
// Both only pay while the loop body is short relative to LDS latency, and the
// body scales with CDIM: -14.5% at CDIM=3 but +2.8% at CDIM=16, so gate at 8.
#if USE_ROCM
#ifndef GSPLAT_FWD_F4_MAXCDIM
#define GSPLAT_FWD_F4_MAXCDIM 8
#endif
#else
// CUDA keeps the stock three-array batch: CDIM is never <= 0, so USE_F4 and the
// host-side shmem sizing below both fall to the original path.
#define GSPLAT_FWD_F4_MAXCDIM 0
#endif

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_fwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2 *__restrict__ means2d,         // [I, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [I, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [I, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [I, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [I, CDIM]
    const bool *__restrict__ masks,           // [I, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,   // tile width in pixels
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int64_t *__restrict__ tile_offsets, // [I, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    scalar_t
        *__restrict__ render_colors, // [I, image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas, // [I, image_height, image_width, 1]
    int32_t *__restrict__ last_ids        // [I, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t image_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    render_colors += image_id * image_height * image_width * CDIM;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int64_t range_start = tile_offsets[tile_id];
    int64_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    constexpr bool USE_F4 = (CDIM <= GSPLAT_FWD_F4_MAXCDIM);
#if USE_ROCM
    // one 16 B-aligned array, two float4 per batch entry (see note at top);
    // the large-CDIM path reinterprets the same slab as the stock three arrays
    extern __shared__ __align__(16) float4 s4raw[];
    float4 *s4 = s4raw;                                  // [2*block_size + 2]
    int32_t *id_batch = (int32_t *)s4raw;                // [block_size]
#else
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                    // [block_size]
    float4 *s4 = nullptr; // never read: USE_F4 is false on the CUDA path
#endif
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]);

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float pix_out[CDIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int64_t batch_start = range_start + block_size * b;
        int64_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            const vec3 cn = conics[g];
            if constexpr (USE_F4) {
                s4[2 * tr] = make_float4(xy.x, xy.y, opac, __int_as_float(g));
                s4[2 * tr + 1] = make_float4(cn.x, cn.y, cn.z, 0.f);
            } else {
                id_batch[tr] = g;
                xy_opacity_batch[tr] = {xy.x, xy.y, opac};
                conic_batch[tr] = cn;
            }
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = (uint32_t)min((int64_t)block_size, range_end - batch_start);
        float4 p0, p1;
        if constexpr (USE_F4) { p0 = s4[0]; p1 = s4[1]; }
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            vec3 conic, xy_opac;
            int32_t g;
            if constexpr (USE_F4) {
                // consume entry t, already in registers, and issue entry t+1 now
                // so its 72-cycle LDS latency overlaps this iteration's arithmetic
                const float4 e0 = p0;
                const float4 e1 = p1;
                p0 = s4[2 * t + 2];
                p1 = s4[2 * t + 3];
                conic = {e1.x, e1.y, e1.z};
                xy_opac = {e0.x, e0.y, e0.z};
                g = __float_as_int(e0.w);
            } else {
                conic = conic_batch[t];
                xy_opac = xy_opacity_batch[t];
                g = id_batch[t];
            }
            const float opac = xy_opac.z;
            const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done: exclusive
                done = true;
                break;
            }

            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,   // tile width in pixels
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders, // [..., image_height, image_width, channels]
    at::Tensor alphas,  // [..., image_height, image_width]
    at::Tensor last_ids // [..., image_height, image_width]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    // CDIM <= GSPLAT_FWD_F4_MAXCDIM: two float4 per batch entry in one
    // 16 B-aligned array, plus one spare entry (32 B) because the loop
    // prefetches entry t+1, so the last prefetch stays in bounds.
    // Larger CDIM, and all of CUDA (where MAXCDIM is 0): the stock three-array
    // layout (see the note at the top of this file for why prefetch stops
    // paying).
    int64_t shmem_size =
        (CDIM <= GSPLAT_FWD_F4_MAXCDIM)
            ? ((int64_t)tile_size * tile_size * 2 + 2) * 4 * sizeof(float)
            : (int64_t)tile_size * tile_size *
                  (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));

#ifndef USE_ROCM
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }
#else
    hipError_t err = hipFuncSetAttribute(
        reinterpret_cast<void*>(rasterize_to_pixels_3dgs_fwd_kernel<CDIM,float>), // Cast to void*
        hipFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_size) // HIP requires int for shared memory size
    );

    if (err != hipSuccess) {
        std::stringstream ss;
        ss << "Failed to set maximum shared memory size (requested " << shmem_size << " bytes), try lowering tile_size.  HIP Error: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
#endif


    rasterize_to_pixels_3dgs_fwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, GET_CURRENT_STREAM()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int64_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_fwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor conics,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        at::Tensor renders,                                                    \
        at::Tensor alphas,                                                     \
        at::Tensor last_ids                                                    \
    );

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
__INS__(64)
__INS__(65)
__INS__(128)
__INS__(129)
__INS__(256)
__INS__(257)
__INS__(512)
__INS__(513)
#undef __INS__

} // namespace gsplat
