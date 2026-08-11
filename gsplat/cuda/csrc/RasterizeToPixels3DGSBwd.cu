#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>


#include "Common.h"
#include "Common.cuh"
#include "Rasterization.h"
#include "Utils.cuh"

#define DEBUG_PRINT 0
#ifdef DEBUG_PRINT
#include <cstdio> // Only include cstdio if DEBUG_PRINT is enabled
#endif

namespace gsplat {

namespace cg = cooperative_groups;


// The wave32 multi-tile backward kernel is gated on CDIM by register pressure,
// not LDS (it uses no dynamic shared memory). VGPRs run at ~69 + 5*CDIM, which
// costs occupancy on gfx1250: CDIM 3 -> 10 waves/SIMD, 8 -> 9, 16 -> 6. The
// hard ceiling is 24, where the lane-scatter static_assert(NGRAD <= 32) fails.
// Larger CDIM falls back to the generic multi-warp kernel.
#ifndef GSPLAT_BS32_MULTITILE_MAXCDIM
#define GSPLAT_BS32_MULTITILE_MAXCDIM 16
#endif

//compiler issue with mov_dpp intrinsic seen in Rocm 6.4.1, so mov_dpp intrinsic is temporarily commented out and replaced with rocprim which also uses dpp when in single wave
// The DPP / "bs64" fast paths below assume a 64-lane wavefront; only compile
// them for wave64 builds (gfx1250 etc. are wave32).
#if USE_ROCM && GSPLAT_USE_WAVE64
template <typename T>
__device__ void dpp_sclr_warpSum(T &val) {
	// T tmp = val + __builtin_amdgcn_mov_dpp(val, 0x118, 0xf, 0xf, 1); //ROW_SHR8
	// tmp = tmp + __builtin_amdgcn_mov_dpp(tmp, 0x114, 0xf, 0xf, 1); //ROW_SHR4
	// tmp = tmp + __builtin_amdgcn_mov_dpp(tmp, 0x112, 0xf, 0xf, 1); //ROW_SHR2
	// tmp = tmp + __builtin_amdgcn_mov_dpp(tmp, 0x111, 0xf, 0xf, 1); //ROW_SHR1
	// tmp = tmp + __builtin_amdgcn_mov_dpp(tmp, 0x142, 0xf, 0xf, 1); //BCAST15
	// tmp = tmp + __builtin_amdgcn_mov_dpp(tmp, 0x143, 0xf, 0xf, 1); //BCAST31
	// val = __shfl(tmp, 63);
    rocprim_warpSum<64>(val, NULL);
}

// This version does reduce but stores the result to a specific location (n_val) on a given lane (ln)
// It can be sued to generate results than can be stored wave-coalesed.
template <typename T>
__device__ void dpp_sprd_warpSum(T &val, int ln, T &n_val) {
	// T tmp = val + __builtin_amdgcn_mov_dpp(val, 0x118, 0xf, 0xf, 1); //ROW_SHR8
	// tmp = tmp + __builtin_amdgcn_mov_dpp(tmp, 0x114, 0xf, 0xf, 1); //ROW_SHR4
	// tmp = tmp + __builtin_amdgcn_mov_dpp(tmp, 0x112, 0xf, 0xf, 1); //ROW_SHR2
	// tmp = tmp + __builtin_amdgcn_mov_dpp(tmp, 0x111, 0xf, 0xf, 1); //ROW_SHR1
	// tmp = tmp + __builtin_amdgcn_mov_dpp(tmp, 0x142, 0xf, 0xf, 1); //BCAST15
	// tmp = tmp + __builtin_amdgcn_mov_dpp(tmp, 0x143, 0xf, 0xf, 1); //BCAST31
	// tmp = __shfl(tmp, 63);
	// if (cg::this_thread_block().thread_rank() == ln)
	//        n_val = tmp;
    T tmp = val;
    rocprim_warpSum<64>(tmp, NULL);
    if (cg::this_thread_block().thread_rank() == ln)
        n_val = tmp;
}

// Vector eltwise reduce, with results spread across lanes of the wave
template <uint32_t numel, typename T>
__device__ void dpp_vec_warpSum(T &val) {
          #pragma unroll
          for (int e=0; e<numel; e++)
            dpp_sprd_warpSum(val[e], e%64, val[e/64]);
}

template <typename T>
__device__ void dpp_warpSum(T &val) {
	if constexpr(std::is_same<T, vec3>::value) {
          dpp_sclr_warpSum(val.x);
          dpp_sclr_warpSum(val.y);
          dpp_sclr_warpSum(val.z);
	}
	else if constexpr(std::is_same<T, vec2>::value) {
          dpp_sclr_warpSum(val.x);
          dpp_sclr_warpSum(val.y);
	}
	else
          dpp_sclr_warpSum(val);
}

template <typename T>
__device__ T dpp_warpMax(T &val) {
	// using ncT = typename std::remove_const<T>::type;
	// ncT tmp = max(val, __builtin_amdgcn_mov_dpp(val, 0x118, 0xf, 0xf, 1)); //ROW_SHR8
	// tmp = max(tmp, __builtin_amdgcn_mov_dpp(tmp, 0x114, 0xf, 0xf, 1)); //ROW_SHR4
	// tmp = max(tmp, __builtin_amdgcn_mov_dpp(tmp, 0x112, 0xf, 0xf, 1)); //ROW_SHR2
	// tmp = max(tmp, __builtin_amdgcn_mov_dpp(tmp, 0x111, 0xf, 0xf, 1)); //ROW_SHR1
	// tmp = max(tmp, __builtin_amdgcn_mov_dpp(tmp, 0x142, 0xf, 0xf, 1)); //BCAST15
	// tmp = max(tmp, __builtin_amdgcn_mov_dpp(tmp, 0x143, 0xf, 0xf, 1)); //BCAST31
	// return __shfl(tmp, 63);
    __shared__ typename rocprim::warp_reduce<int32_t, 64>::storage_type warp_storage;
    rocprim::warp_reduce<int32_t, 64> wreduce;
    int32_t max;
    wreduce.reduce(val,            // 1) value held by this lane
            max,               // 2) reference that will receive the result
            warp_storage,                 // 3) shared-memory storage
            rocprim::maximum<int32_t>());
    return max;
}

template <uint32_t CDIM, typename scalar_t>
__launch_bounds__(64)
__global__ void rasterize_bs64_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,   // tile width in pixels
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t
        *__restrict__ render_alphas,      // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [..., image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [..., image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [..., N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,       // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities, // [..., N] or [nnz]
    const uint32_t max_batch_size
) {
    auto block = cg::this_thread_block();
    uint32_t image_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();

    const uint32_t batch_allocation_size = max_batch_size;
    const uint32_t num_batches =
        (range_end - range_start + max_batch_size - 1) / max_batch_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [batch_allocation_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[batch_allocation_size]); // [batch_allocation_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[batch_allocation_size]); // [batch_allocation_size]
    float *rgbs_batch =
        (float *)s; // [batch_allocation_size * CDIM]

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();

    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);

    int32_t warp_bin_final =
    dpp_warpMax(bin_final);
    int32_t _id_batch;
    vec3 _xy_opacity_batch;
    vec3 _conic_batch;
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int64_t batch_end = range_end - 1 - max_batch_size * b;
        const uint32_t current_batch_size = (uint32_t)min((int64_t)max_batch_size, batch_end + 1 - range_start);
        const int64_t idx = batch_end - tr;
        if (tr < current_batch_size && idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            _id_batch = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            _xy_opacity_batch = {xy.x, xy.y, opac};
            _conic_batch = conics[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < current_batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            float vis;
            conic.x = __shfl(_conic_batch.x, t);
            conic.y = __shfl(_conic_batch.y, t);
            conic.z = __shfl(_conic_batch.z, t);
            vec3 xy_opac;
            xy_opac.x = __shfl(_xy_opacity_batch.x, t);
            xy_opac.y = __shfl(_xy_opacity_batch.y, t);
            xy_opac.z = __shfl(_xy_opacity_batch.z, t);
            if (valid) {
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                      conic.z * delta.y * delta.y) +
                              conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            dpp_vec_warpSum<CDIM>(v_rgb_local);   // CDIM-sized float array
            dpp_warpSum(v_conic_local); // float
            dpp_warpSum(v_xy_local);    // vec2
            if (v_means2d_abs != nullptr)
                dpp_warpSum(v_xy_abs_local);// vec2
            dpp_warpSum(v_opacity_local);// float
	    int32_t g = __shfl(_id_batch, t); // flatten index in [I * N] or [nnz]

            float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; k+=64) {
		if (k + warp.thread_rank() < CDIM)
                    atomicAdd(v_rgb_ptr + k + warp.thread_rank(), v_rgb_local[k/64]);
            }

            if (warp.thread_rank() == 0) {
                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                atomicAdd(v_conic_ptr, v_conic_local.x);
                atomicAdd(v_conic_ptr + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    atomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    atomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                atomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

#endif

// ---------------------------------------------------------------------------
// Wave32 (RDNA / CDNA5, e.g. gfx1250) backward kernel: the wave32 analog of the
// wave64 "bs64" kernel. The launcher uses it for the square (block_size==64,
// 8x8) tile with CDIM <= GSPLAT_BS32_MULTITILE_MAXCDIM; every other tile shape
// and larger CDIM go to the generic multi-warp kernel.
// All reductions use rocprim::warp_reduce<T,32> (DPP) to match the ROCm path.
#if USE_ROCM && !GSPLAT_USE_WAVE64

// ---------------------------------------------------------------------------
// MULTI-TILE: a 256-thread block == 8 wave32 waves, where EACH wave
// independently rasterizes a DIFFERENT 8x8 tile with a register+shfl geometry:
// 2 px/lane, per-wave rocprim (DPP) reduction, one atomicAdd set per Gaussian
// per tile. The 8 tiles handled by a block are (blockIdx.y * MT_WARPS +
// warp_id). Because the waves own disjoint tiles, there is NO cross-wave
// coupling: no block.sync(), no shared batch, and divergent per-wave loop trip
// counts / early returns are all safe. The point is to supply 8 resident waves
// per workgroup (deep occupancy, like tile16) while keeping the 8x8 tile
// granularity of bs32. The per-wave rocprim scratch is replicated MT_WARPS
// times and indexed by warp_id.
//
// H8: ABSGRAD is a compile-time flag for the v_means2d_abs path. When the caller
// does not request abs gradients (absgrad=false, the common case) the launcher
// instantiates ABSGRAD=false, so the abs accumulator, its warp reduction and its
// atomics are compiled out entirely (no dead registers / VALU / atomics).
template <uint32_t CDIM, typename scalar_t, bool ABSGRAD>
__launch_bounds__(256)
__global__ void rasterize_bs32_8tile_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size, // tile extent in pixels (tiles are square)
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t
        *__restrict__ render_alphas,      // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [..., image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [..., image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [..., N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,       // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities, // [..., N] or [nnz]
    const uint32_t max_batch_size
) {
    auto block = cg::this_thread_block();
    constexpr uint32_t MT_WARPS = 256u / GSPLAT_WARP_SIZE; // tiles per block (8)

    cg::thread_block_tile<GSPLAT_WARP_SIZE> warp =
        cg::tiled_partition<GSPLAT_WARP_SIZE>(block);
    const uint32_t lane = warp.thread_rank();
    const uint32_t warp_id = block.thread_rank() / GSPLAT_WARP_SIZE; // 0..MT_WARPS-1

    const uint32_t image_id = block.group_index().x;
    const uint32_t total_tiles = tile_width * tile_height;
    const uint32_t tile_linear = block.group_index().y * MT_WARPS + warp_id;

    tile_offsets += image_id * tile_height * tile_width;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    // Per-warp rocprim scratch: one storage object per wave, indexed by warp_id
    // (rocprim_warpSum / bin_reduce use warp_storage_base[warp_id] internally).
    __shared__ typename rocprim::warp_reduce<int32_t, GSPLAT_WARP_SIZE>::storage_type
        bin_storage[MT_WARPS];
    __shared__ typename rocprim::warp_reduce<float, GSPLAT_WARP_SIZE>::storage_type
        sum_storage[MT_WARPS];

    // Guard: a wave mapped past the last tile does nothing. Safe because there
    // are NO block-wide barriers in this kernel (only warp-scoped ops), so a
    // whole-wave early return cannot deadlock the block. tile_linear is uniform
    // across the wave (depends only on warp_id), and so is the mask test.
    if (tile_linear >= total_tiles) {
        return;
    }
    const uint32_t tile_id = tile_linear;
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }
    const uint32_t trow = tile_linear / tile_width; // tile row in the tile grid
    const uint32_t tcol = tile_linear % tile_width; // tile col in the tile grid

    // Each lane owns 2 pixels of this wave's 8x8 tile: p = lane + 32*s, s in {0,1}.
    float px[2], py[2];
    float T[2]; // T_final / v_render_alpha live in r_tf / r_va below
    int32_t bin_final_arr[2];

    // NO dynamic LDS in the inner loop.
    //
    // This kernel used to stage four arrays in shared memory -- s_vrc
    // (v_render_colors), s_tf (T_final), s_va (v_render_alpha) and s_buf (the
    // colour accumulator). All four are indexed ONLY by this lane's own pixel
    // (p = lane + 32*s); none is ever read across lanes, so LDS bought nothing
    // but latency. Worse, the compiler could not prove the s_buf stores did not
    // alias the others (all were offsets into the same `extern __shared__`
    // array), so the loop-INVARIANT reads were re-issued every inner iteration.
    //
    // They now live in registers (r_vrc / r_tf / r_va below), the colour
    // accumulator collapses to the scalar r_sdot, and Gaussian colours are
    // broadcast with warp.shfl exactly as conic and xy/opacity already were.
    // Measured: LDS instructions 46 -> 12, SQ_INSTS_LDS -39.7%, SQ_WAIT_ANY
    // -23.8%, kernel 2.937 -> 2.541 ms mean (-13.5%). Memory traffic is
    // unchanged to the byte -- this moved where data lives, not how much moves.
    float r_vrc[2][CDIM];
    float r_tf[2], r_va[2];
    // ---- 4C: collapse the CDIM-vector colour accumulator to ONE scalar ----
    //
    // `buf_k` accumulates sum_over_processed_gaussians rgb_k * fac, and is only
    // ever consumed dotted against vrc_k, which is CONSTANT per pixel:
    //
    //     v_alpha += sum_k (rgb_k * T - buf_k * ra) * vrc_k
    //              = T * sum_k rgb_k*vrc_k  -  ra * sum_k buf_k*vrc_k
    //              = T * R                  -  ra * S          , R := sum_k rgb_k*vrc_k
    //     buf_k += rgb_k * fac   =>   S += fac * R
    //
    // So the whole CDIM-length accumulator collapses to the scalar S, and the
    // per-(pixel, Gaussian) work drops from ~12 VALU ops to ~6. Costs 2 VGPRs
    // instead of 2*CDIM, and removes s_buf/r_buf entirely.
    //
    // NOT bit-identical: the summation is reassociated, so this must clear the
    // golden-gradient tolerance gate rather than match exactly.
    float r_sdot[2] = {0.f, 0.f};

#pragma unroll
    for (int s = 0; s < 2; ++s) {
        uint32_t p = lane + 32u * (uint32_t)s; // 0..63 within the tile
        uint32_t row = p / tile_size;
        uint32_t col = p % tile_size;
        uint32_t i = trow * tile_size + row;
        uint32_t j = tcol * tile_size + col;
        px[s] = (float)j + 0.5f;
        py[s] = (float)i + 0.5f;
        // H1: pix_id is dead after this loop -> loop-local, not a kernel-wide array.
        const int32_t pix_id =
            min(i * image_width + j, image_width * image_height - 1);
        const bool inside = (i < image_height && j < image_width);
        const float tf_h3 = 1.0f - render_alphas[pix_id];
        r_tf[s] = tf_h3;
        T[s] = tf_h3;
        // H2: outside pixels get bin_final = -1. The inner-loop test
        // (batch_end - t <= bin_final_arr[s]) is then always false for them,
        // so a separate kernel-wide inside_arr[2] is no longer needed.
        bin_final_arr[s] = inside ? last_ids[pix_id] : -1;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            /* r_sdot replaces the whole buf accumulator */
            r_vrc[s][k] = v_render_colors[pix_id * CDIM + k];
        }
        r_va[s] = v_render_alphas[pix_id];
    }

    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];


    const uint32_t num_batches =
        (range_end - range_start + max_batch_size - 1) / max_batch_size;

    // furthest gaussian any of this wave's 64 pixels needs
    int32_t local_bin_final = max(bin_final_arr[0], bin_final_arr[1]);
    rocprim::warp_reduce<int32_t, GSPLAT_WARP_SIZE> bin_reduce;
    int32_t warp_bin_final;
    bin_reduce.reduce(
        local_bin_final, warp_bin_final, bin_storage[warp_id],
        rocprim::maximum<int32_t>());

    int32_t _id_batch;
    vec3 _xy_opacity_batch;
    vec3 _conic_batch;
    // Colours held per-lane and broadcast with warp.shfl, exactly as conic and
    // xy_opacity already are, instead of round-tripping through LDS. Removes
    // the last genuinely shared LDS array from the inner loop.
    float _rgb_batch[CDIM];
    for (uint32_t b = 0; b < num_batches; ++b) {
        warp.sync();

        // one wave of 32 lanes loads 32 gaussians (each once, no duplication)
        const int64_t batch_end = range_end - 1 - max_batch_size * b;
        const uint32_t current_batch_size =
            (uint32_t)min((int64_t)max_batch_size, batch_end + 1 - range_start);
        const int64_t idx = batch_end - lane;
        if (lane < current_batch_size && idx >= range_start) {
            int32_t g = flatten_ids[idx];
            _id_batch = g;
            const int32_t gl = g;
            const int32_t gp = gl;
            const vec2 xy = means2d[gp];
            const float opac = opacities[gp];
            _xy_opacity_batch = {xy.x, xy.y, opac};
            _conic_batch = conics[gp];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                _rgb_batch[k] = colors[gp * CDIM + k];
            }
        }
        warp.sync();

        for (uint32_t t = max(0, batch_end - warp_bin_final);
             t < current_batch_size; ++t) {
            // broadcast gaussian t (held by lane t) to every lane in the wave
            vec3 conic;
            conic.x = warp.shfl(_conic_batch.x, t);
            conic.y = warp.shfl(_conic_batch.y, t);
            conic.z = warp.shfl(_conic_batch.z, t);
            vec3 xy_opac;
            xy_opac.x = warp.shfl(_xy_opacity_batch.x, t);
            xy_opac.y = warp.shfl(_xy_opacity_batch.y, t);
            xy_opac.z = warp.shfl(_xy_opacity_batch.z, t);
            float rgb_t[CDIM];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k)
                rgb_t[k] = warp.shfl(_rgb_batch[k], t);

            bool valid_s[2];
            float alpha_s[2];
            float vis_s[2];
            vec2 delta_s[2];
            bool any_valid = false;

            // Phase 1: evaluate both of this lane's pixels in one basic block.
            // The gaussian test is pure arithmetic on values that are always
            // finite (px/py and the clamped loads are initialized for outside
            // pixels too), so running it for lanes that turn out invalid is
            // safe and costs only VALU.
            //
            // The point is the two __expf chains. Guarding each one with its
            // own `if` writes EXEC between them, which forces the scheduler to
            // keep the s=0 and s=1 transcendentals in separate regions and
            // serializes two dependency chains that have no data dependence on
            // each other. In one region they interleave.
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                delta_s[s] = {xy_opac.x - px[s], xy_opac.y - py[s]};
                float sigma =
                    0.5f * (conic.x * delta_s[s].x * delta_s[s].x +
                            conic.z * delta_s[s].y * delta_s[s].y) +
                    conic.y * delta_s[s].x * delta_s[s].y;
                vis_s[s] = __expf(-sigma);
                alpha_s[s] = min(0.999f, xy_opac.z * vis_s[s]);
                // Bitwise & on purpose: && short-circuits, and each short
                // circuit is another EXEC write in the middle of phase 1.
                // H2: outside pixels have bin_final_arr[s] == -1, so the first
                // term is false for them (batch_end - t >= 0).
                valid_s[s] = (batch_end - t <= bin_final_arr[s]) &
                             !(sigma < 0.f) & !(alpha_s[s] < ALPHA_THRESHOLD);
                any_valid |= valid_s[s];
            }

            // Nothing in this wave touches gaussian t: skip the gradient
            // bodies and the (relatively expensive) reduction alike.
            if (!warp.any(any_valid)) {
                continue;
            }

            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;

            // Phase 2: the gradient bodies. These read and write per-pixel
            // state (T[s], r_sdot[s]) so they keep one EXEC region each.
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                if (valid_s[s]) {
                    const float alpha = alpha_s[s];
                    const float vis = vis_s[s];
                    const vec2 delta = delta_s[s];
                    const float opac_s = xy_opac.z;
                    const float *vrc_s = r_vrc[s];
                    float ra = 1.0f / (1.0f - alpha);
                    T[s] *= ra;
                    const float fac = alpha * T[s];
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_rgb_local[k] += fac * vrc_s[k];
                    }
                    float v_alpha = 0.f;
                    float R_dot = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        R_dot += rgb_t[k] * vrc_s[k];
                    }
                    v_alpha += R_dot * T[s] - r_sdot[s] * ra;
                    v_alpha += r_tf[s] * ra * r_va[s];
                    if (backgrounds != nullptr) {
                        float accum = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            accum += backgrounds[k] * vrc_s[k];
                        }
                        v_alpha += -r_tf[s] * ra * accum;
                    }
                    if (opac_s * vis <= 0.999f) {
                        const float v_sigma = -opac_s * vis * v_alpha;
                        v_conic_local.x += 0.5f * v_sigma * delta.x * delta.x;
                        v_conic_local.y += v_sigma * delta.x * delta.y;
                        v_conic_local.z += 0.5f * v_sigma * delta.y * delta.y;
                        float vxl =
                            v_sigma * (conic.x * delta.x + conic.y * delta.y);
                        float vyl =
                            v_sigma * (conic.y * delta.x + conic.z * delta.y);
                        v_xy_local.x += vxl;
                        v_xy_local.y += vyl;
                        if constexpr (ABSGRAD) {
                            v_xy_abs_local.x += abs(vxl);
                            v_xy_abs_local.y += abs(vyl);
                        }
                        v_opacity_local += vis * v_alpha;
                    }
                    r_sdot[s] += R_dot * fac;
                }
            }

            // single 32-lane rocprim (DPP) reduction across this wave's tile
            rocprim_warpSum<CDIM, GSPLAT_WARP_SIZE>(v_rgb_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_conic_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_local, sum_storage);
            if constexpr (ABSGRAD)
                rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_abs_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_opacity_local, sum_storage);

            int32_t g = warp.shfl(_id_batch, t);
            // LANE-SCATTER epilogue.
            //
            // The conventional epilogue (still used by the generic kernel below)
            // leaves all NGRAD reduced components in lane 0, which then issues
            // NGRAD *serialized* atomicAdds -- the wave stalls on each one's
            // issue slot in turn. The win grows with CDIM: NGRAD is 9 at CDIM=3
            // but 22 at CDIM=16. Here we instead give each of lanes [0, NGRAD)
            // one component and one destination address, so the wave issues a
            // SINGLE global_atomic_add_f32 with NGRAD lanes
            // active. Same number of atomic values, same arithmetic (bitwise
            // identical: each component is still one f32 atomic add to the same
            // address), but the issue is parallel across lanes instead of serial.
            //
            // Lane map (CDIM=3, ABSGRAD=0 -> NGRAD=9):
            //   [0, CDIM)          -> v_colors [CDIM*g + k]
            //   CDIM+0 .. CDIM+2   -> v_conics [3*g + 0..2]
            //   CDIM+3 .. CDIM+4   -> v_means2d[2*g + 0..1]
            //   CDIM+5             -> v_opacities[g]
            //   CDIM+6 .. CDIM+7   -> v_means2d_abs[2*g + 0..1]   (ABSGRAD only)
            constexpr uint32_t NGRAD = CDIM + 6u + (ABSGRAD ? 2u : 0u);
            static_assert(NGRAD <= GSPLAT_WARP_SIZE,
                          "lane-scatter needs one lane per gradient component");

            // Every participating lane must hold the reduced sums. The DPP
            // all-reduce (dpp_permlane_warpSum32) already leaves them in every
            // lane; rocprim::warp_reduce defaults to UseAllReduce=false and
            // leaves them in lane 0 only, so broadcast in that case.
            float *ls_ptr = nullptr;
            float ls_val = 0.f;
            // Comparisons are against compile-time constants so each becomes a
            // lane-mask select; the arrays stay in registers.
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                if (lane == k) {
                    ls_ptr = (float *)(v_colors) + CDIM * g + k;
                    ls_val = v_rgb_local[k];
                }
            }
            if (lane == CDIM + 0u) { ls_ptr = (float *)(v_conics) + 3 * g + 0; ls_val = v_conic_local.x; }
            if (lane == CDIM + 1u) { ls_ptr = (float *)(v_conics) + 3 * g + 1; ls_val = v_conic_local.y; }
            if (lane == CDIM + 2u) { ls_ptr = (float *)(v_conics) + 3 * g + 2; ls_val = v_conic_local.z; }
            if (lane == CDIM + 3u) { ls_ptr = (float *)(v_means2d) + 2 * g + 0; ls_val = v_xy_local.x; }
            if (lane == CDIM + 4u) { ls_ptr = (float *)(v_means2d) + 2 * g + 1; ls_val = v_xy_local.y; }
            if (lane == CDIM + 5u) { ls_ptr = v_opacities + g;                  ls_val = v_opacity_local; }
            if constexpr (ABSGRAD) {
                if (lane == CDIM + 6u) { ls_ptr = (float *)(v_means2d_abs) + 2 * g + 0; ls_val = v_xy_abs_local.x; }
                if (lane == CDIM + 7u) { ls_ptr = (float *)(v_means2d_abs) + 2 * g + 1; ls_val = v_xy_abs_local.y; }
            }
            if (ls_ptr != nullptr) {
                atomicAdd(ls_ptr, ls_val);
            }
        }
    }
}

#endif

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,   // tile width in pixels
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t
        *__restrict__ render_alphas,      // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [..., image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [..., image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [..., N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,       // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities, // [..., N] or [nnz]
    const uint32_t max_batch_size
) {
    auto block = cg::this_thread_block();
    uint32_t image_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int64_t range_start = tile_offsets[tile_id];
    int64_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();

#if USE_ROCM
    const uint32_t batch_allocation_size = max_batch_size;
    const uint32_t num_batches =
        (range_end - range_start + max_batch_size - 1) / max_batch_size;
#else
    const uint32_t batch_allocation_size = block_size;
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;
#endif

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [batch_allocation_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[batch_allocation_size]); // [batch_allocation_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[batch_allocation_size]); // [batch_allocation_size]
    float *rgbs_batch =
        (float *)&conic_batch[batch_allocation_size]; // [batch_allocation_size * CDIM]
    
    #if USE_ROCM
    using warp_reduce_float_t = rocprim::warp_reduce<float,GSPLAT_WARP_SIZE>;
    auto* warp_storage_base   =
    (typename warp_reduce_float_t::storage_type*)
        (rgbs_batch + batch_allocation_size * CDIM);
    #endif

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    
    #if USE_ROCM
    cg::thread_block_tile<GSPLAT_WARP_SIZE> warp = cg::tiled_partition<GSPLAT_WARP_SIZE>(block);
    #else
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    #endif
    
    #if USE_ROCM
        __shared__ typename rocprim::warp_reduce<int32_t, GSPLAT_WARP_SIZE>::storage_type warp_storage;
        rocprim::warp_reduce<int32_t, GSPLAT_WARP_SIZE> wreduce;
        int32_t warp_bin_final;
        wreduce.reduce( bin_final,            // 1) value held by this lane
                warp_bin_final,               // 2) reference that will receive the result
                warp_storage,                 // 3) shared-memory storage
                rocprim::maximum<int32_t>()); // 4) binary operator
    #else
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
    #endif
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
#if USE_ROCM
        const int64_t batch_end = range_end - 1 - max_batch_size * b;
        const uint32_t current_batch_size = (uint32_t)min((int64_t)max_batch_size, batch_end + 1 - range_start);
        const int64_t idx = batch_end - tr;
        if (tr < current_batch_size && idx >= range_start) {
#else
        const int64_t batch_end = range_end - 1 - block_size * b;
        const uint32_t batch_size = (uint32_t)min((int64_t)block_size, batch_end + 1 - range_start);
        const int64_t idx = batch_end - tr;
        if (idx >= range_start) {
#endif
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
#if USE_ROCM
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < current_batch_size; ++t) {
#else
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
#endif
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            float vis;

            if (valid) {
                conic = conic_batch[t];
                vec3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                      conic.z * delta.y * delta.y) +
                              conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            #if USE_ROCM
            rocprim_warpSum<CDIM, GSPLAT_WARP_SIZE>(v_rgb_local, warp_storage_base);   // CDIM-sized float array
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_conic_local, warp_storage_base); // float
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_local, warp_storage_base);    // vec2
            if (v_means2d_abs != nullptr)
                rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_abs_local, warp_storage_base);// vec2
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_opacity_local, warp_storage_base);// float
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    atomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                atomicAdd(v_conic_ptr, v_conic_local.x);
                atomicAdd(v_conic_ptr + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    atomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    atomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                atomicAdd(v_opacities + g, v_opacity_local);
            }
            #else
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
            #endif
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,   // tile width in pixels
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = render_alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

#if USE_ROCM
    // Optimization for ROCm: Use smaller batch size to reduce shared memory usage

    const uint32_t block_size = tile_size * tile_size;
    uint32_t max_batch_size;
    int64_t shmem_size;
#if GSPLAT_USE_WAVE64
    if (block_size == 64) { // wave64-optimized path
      max_batch_size = 32;
      //max_batch_size = min(max_batch_size, block_size);
      if (CDIM <= 32) {
        max_batch_size = block_size;
      }
      shmem_size =
        max_batch_size *
        (sizeof(float) * CDIM);
    } else
#else
    // wave32 multi-tile register+shfl path: a 256-thread block (MT_WARPS == 8
    // wave32 waves) where each wave independently rasterizes ONE 8x8 tile with
    // 2 pixels per lane, and the grid packs MT_WARPS tiles per block. Gated on
    // CDIM by register pressure (see GSPLAT_BS32_MULTITILE_MAXCDIM); larger CDIM
    // and every other tile shape use the generic multi-warp config below.
    if (block_size == 64 && CDIM <= GSPLAT_BS32_MULTITILE_MAXCDIM) {
      constexpr uint32_t MT_WARPS = 256u / GSPLAT_WARP_SIZE; // 8
      max_batch_size = GSPLAT_WARP_SIZE; // 32, each wave loads its own batch
      // Zero dynamic LDS: every per-pixel array this kernel used to stage in
      // shared memory now lives in registers, and Gaussian colours are
      // broadcast with warp.shfl. The only LDS left is the static rocprim
      // scratch, which is empty on wave32 since DPP needs no storage -- builds
      // report 0 bytes/block for every instantiation.
      shmem_size = 0;
      threads = dim3{256, 1, 1};
      grid = dim3{I, (tile_height * tile_width + MT_WARPS - 1) / MT_WARPS, 1};
    } else
#endif
    {
      max_batch_size = 16;
      max_batch_size = min(max_batch_size, block_size);
      if (CDIM <= 16) {
        max_batch_size = block_size;
      }
      const uint32_t warps_per_block = (block_size + GSPLAT_WARP_SIZE - 1) / GSPLAT_WARP_SIZE;
      std::size_t warp_scratch_bytes =
        warps_per_block * sizeof(typename rocprim::warp_reduce<float,GSPLAT_WARP_SIZE>::storage_type);
      shmem_size =
        max_batch_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM) + warp_scratch_bytes;
    }
#else
    // Original CUDA implementation
    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);
#endif

    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    
 #ifndef USE_ROCM
    auto KERNEL = rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>;
    if (cudaFuncSetAttribute(
            KERNEL,
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
    auto KERNEL = rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>;
#if GSPLAT_USE_WAVE64
    if (block_size == 64) {
        KERNEL = rasterize_bs64_to_pixels_3dgs_bwd_kernel<CDIM, float>;
    }
#else
    // Wave32: default to the generic multi-warp kernel. The square 8x8
    // (block_size==64) tile with small CDIM uses the multi-tile kernel instead.
    // All other tile sizes (incl. 32) and larger CDIM fall back to the generic
    // kernel.
    if constexpr (CDIM <= GSPLAT_BS32_MULTITILE_MAXCDIM) {
        if (block_size == 64) {
            // H8: pick the ABSGRAD instantiation so the abs path is compiled
            // out when the caller does not request abs gradients.
            KERNEL = v_means2d_abs.has_value()
                ? rasterize_bs32_8tile_to_pixels_3dgs_bwd_kernel<CDIM, float, true>
                : rasterize_bs32_8tile_to_pixels_3dgs_bwd_kernel<CDIM, float, false>;
        }
    }
#endif
    hipError_t err = hipFuncSetAttribute(
        reinterpret_cast<void*>(KERNEL), // Cast to void*
        hipFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_size) // HIP requires int for shared memory size
    );

    if (err != hipSuccess) {
        std::stringstream ss;
        ss << "Failed to set maximum shared memory size (requested " << shmem_size << " bytes), try lowering tile_size.  HIP Error: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
#endif

    KERNEL
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
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            max_batch_size
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_bwd_kernel<CDIM>(            \
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
        const at::Tensor render_alphas,                                        \
        const at::Tensor last_ids,                                             \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        at::optional<at::Tensor> v_means2d_abs,                                \
        at::Tensor v_means2d,                                                  \
        at::Tensor v_conics,                                                   \
        at::Tensor v_colors,                                                   \
        at::Tensor v_opacities                                                 \
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

