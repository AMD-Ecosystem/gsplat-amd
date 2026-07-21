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

// Wave32 backward kernel selection for the square (block_size==64, 8x8) tile.
// When 1, use the 2-wave cooperative kernel (64-thread block, 1 px/lane, one
// atomicAdd set per Gaussian per tile via a cross-wave LDS combine). When 0,
// fall back to the 1-wave / 2-px-per-lane kernel. Kept as a compile-time toggle
// for A/B benchmarking on wave32 (gfx1250) parts.
#ifndef GSPLAT_BS32_2WAVE_COOP
#define GSPLAT_BS32_2WAVE_COOP 1
#endif

// Wave32 multi-tile backward kernel (opt-in, off by default). When 1, the
// square (block_size==64, 8x8) tile path launches a 256-thread block (8 wave32
// waves) where EACH wave independently rasterizes one 8x8 tile (bs32_1wave
// body, register+shfl, 2 px/lane, its own reduction + 1 atomic/Gaussian/tile).
// This gives 8 waves/workgroup (deep occupancy, like tile16) WITHOUT any
// cross-wave sync, while keeping the 8x8 tile granularity. Gated to small CDIM
// (<= GSPLAT_BS32_MULTITILE_MAXCDIM) so the 8x per-warp LDS slabs fit in 64KB;
// larger CDIM falls back to the coop / 1-wave kernel.
#ifndef GSPLAT_BS32_MULTITILE
#define GSPLAT_BS32_MULTITILE 0
#endif
#ifndef GSPLAT_BS32_MULTITILE_MAXCDIM
#define GSPLAT_BS32_MULTITILE_MAXCDIM 8
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
    const uint32_t tile_size_h, // tile height in pixels
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
    uint32_t i = block.group_index().y * tile_size_h + block.thread_index().y;
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
// Wave32 (RDNA / CDNA5, e.g. gfx1250) backward kernels: the wave32 analogs of
// the wave64 "bs64" kernel. The active 8x8-tile variants are selected by the
// launcher via the GSPLAT_BS32_2WAVE_COOP toggle:
//   * rasterize_bs32_2wave_coop_... : 64-thread block (two 32-lane waves),
//        1 px/lane, cross-wave LDS combine -> 1 atomic/Gaussian/tile.
//   * rasterize_bs32_1wave_...      : single 32-lane wave, 2 px/lane.
// All reductions use rocprim::warp_reduce<T,32> (DPP) to match the ROCm path.
#if USE_ROCM && !GSPLAT_USE_WAVE64

// ---------------------------------------------------------------------------
// Wave32 (gfx1250) 1-wave variant: a SINGLE 32-lane wave covers the whole 8x8
// (64-pixel) tile with TWO pixels per lane (p = lane + 32*s, s in {0,1}). All
// per-Gaussian attributes (id, mean, opacity, conic AND colors) live in per-lane
// registers and are broadcast within the wave via warp.shfl(); there is no
// dynamic shared memory (only a tiny static rocprim scratch). For each Gaussian
// each lane first sums the gradients of its two pixels in registers, then ONE
// 32-lane rocprim (DPP) reduction collapses across lanes, and lane 0 does a
// SINGLE atomicAdd set per Gaussian per tile. This is the true wave32 analog of
// the wave64 bs64 kernel: 1 atomic/Gaussian/tile and Gaussians loaded once.
//
// Per-pixel state (T, buffer[CDIM], v_render_c[CDIM], ...) is duplicated (x2)
// vs the 1-pixel kernels, so it is gated to small CDIM in the launcher and uses
// __launch_bounds__(32) (32-thread block) which lets the compiler give each lane
// more registers to absorb that doubled state.
template <uint32_t CDIM, typename scalar_t>
__launch_bounds__(32)
__global__ void rasterize_bs32_1wave_to_pixels_3dgs_bwd_kernel(
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
    const uint32_t tile_size_h, // tile height in pixels
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

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const uint32_t lane = warp.thread_rank();

    // Each lane owns 2 pixels of the 8x8 tile: p = lane + 32*s, s in {0,1}.
    // Keeping the two 32-pixel slabs contiguous matches the coalesced global
    // access pattern of the 1-pixel kernels.
    float px[2], py[2];
    int32_t pix_id_arr[2];
    bool inside_arr[2];
    float T[2], T_final[2], v_render_a_arr[2];
    int32_t bin_final_arr[2];

    // Dynamic LDS layout (single 32-lane wave covering a 64-pixel tile):
    //   s_rgbs : [max_batch_size * CDIM]          color batch (shared across lanes)
    //   s_vrc  : [2 * GSPLAT_WARP_SIZE * CDIM]     per-pixel v_render_colors grad
    //   s_buf  : [2 * GSPLAT_WARP_SIZE * CDIM]     per-pixel color accumulator
    // s_vrc/s_buf are per-lane-private (each lane only touches its own two pixels
    // p = lane + 32*s), so they need no cross-lane sync; they live in LDS purely
    // to cut VGPR pressure.
    extern __shared__ float s_mem[];
    float *s_rgbs = s_mem;
    float *s_vrc = s_mem + (size_t)max_batch_size * CDIM;
    float *s_buf = s_vrc + (size_t)2 * GSPLAT_WARP_SIZE * CDIM;

#pragma unroll
    for (int s = 0; s < 2; ++s) {
        uint32_t p = lane + 32u * (uint32_t)s; // 0..63 within the tile
        uint32_t row = p / tile_size;
        uint32_t col = p % tile_size;
        uint32_t i = block.group_index().y * tile_size_h + row;
        uint32_t j = block.group_index().z * tile_size + col;
        px[s] = (float)j + 0.5f;
        py[s] = (float)i + 0.5f;
        pix_id_arr[s] =
            min(i * image_width + j, image_width * image_height - 1);
        inside_arr[s] = (i < image_height && j < image_width);
        T_final[s] = 1.0f - render_alphas[pix_id_arr[s]];
        T[s] = T_final[s];
        bin_final_arr[s] = inside_arr[s] ? last_ids[pix_id_arr[s]] : 0;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            s_buf[p * CDIM + k] = 0.f;
            s_vrc[p * CDIM + k] = v_render_colors[pix_id_arr[s] * CDIM + k];
        }
        v_render_a_arr[s] = v_render_alphas[pix_id_arr[s]];
    }

    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];

    const uint32_t num_batches =
        (range_end - range_start + max_batch_size - 1) / max_batch_size;

    // tiny static rocprim scratch (single wave -> one storage object). This is
    // static shared memory, not the dynamic shared batch buffers we avoid.
    __shared__ typename rocprim::warp_reduce<int32_t, GSPLAT_WARP_SIZE>::storage_type
        bin_storage[1];
    __shared__ typename rocprim::warp_reduce<float, GSPLAT_WARP_SIZE>::storage_type
        sum_storage[1];

    // furthest gaussian any of this wave's 64 pixels needs
    int32_t local_bin_final = max(bin_final_arr[0], bin_final_arr[1]);
    rocprim::warp_reduce<int32_t, GSPLAT_WARP_SIZE> bin_reduce;
    int32_t warp_bin_final;
    bin_reduce.reduce(
        local_bin_final, warp_bin_final, bin_storage[0],
        rocprim::maximum<int32_t>());

    int32_t _id_batch;
    vec3 _xy_opacity_batch;
    vec3 _conic_batch;
    // s_rgbs (colors batch) is staged in dynamic LDS like the bs64 kernel: the
    // single wave writes lane -> [lane*CDIM] and reads Gaussian t from [t*CDIM].
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
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            _xy_opacity_batch = {xy.x, xy.y, opac};
            _conic_batch = conics[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                s_rgbs[lane * CDIM + k] = colors[g * CDIM + k];
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
            // colors for gaussian t are read directly from LDS (s_rgbs) below

            // this lane's contribution for gaussian t, summed over its 2 pixels
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            bool any_valid = false;

#pragma unroll
            for (int s = 0; s < 2; ++s) {
                bool valid = inside_arr[s];
                if (batch_end - t > bin_final_arr[s]) {
                    valid = false;
                }
                float alpha;
                float opac_s;
                float vis;
                vec2 delta;
                if (valid) {
                    opac_s = xy_opac.z;
                    delta = {xy_opac.x - px[s], xy_opac.y - py[s]};
                    float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                          conic.z * delta.y * delta.y) +
                                  conic.y * delta.x * delta.y;
                    vis = __expf(-sigma);
                    alpha = min(0.999f, opac_s * vis);
                    if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                        valid = false;
                    }
                }
                if (valid) {
                    // this lane's per-pixel LDS slabs for pixel s
                    const uint32_t p_s = lane + GSPLAT_WARP_SIZE * (uint32_t)s;
                    const float *vrc_s = s_vrc + p_s * CDIM;
                    float *buf_s = s_buf + p_s * CDIM;
                    float ra = 1.0f / (1.0f - alpha);
                    T[s] *= ra;
                    const float fac = alpha * T[s];
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_rgb_local[k] += fac * vrc_s[k];
                    }
                    float v_alpha = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_alpha += (s_rgbs[t * CDIM + k] * T[s] - buf_s[k] * ra) *
                                   vrc_s[k];
                    }
                    v_alpha += T_final[s] * ra * v_render_a_arr[s];
                    if (backgrounds != nullptr) {
                        float accum = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            accum += backgrounds[k] * vrc_s[k];
                        }
                        v_alpha += -T_final[s] * ra * accum;
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
                        if (v_means2d_abs != nullptr) {
                            v_xy_abs_local.x += abs(vxl);
                            v_xy_abs_local.y += abs(vyl);
                        }
                        v_opacity_local += vis * v_alpha;
                    }
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        buf_s[k] += s_rgbs[t * CDIM + k] * fac;
                    }
                }
                any_valid |= valid;
            }

            // skip the (relatively expensive) reduction if no pixel is active
            if (!warp.any(any_valid)) {
                continue;
            }

            // single 32-lane rocprim (DPP) reduction across the whole tile
            rocprim_warpSum<CDIM, GSPLAT_WARP_SIZE>(v_rgb_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_conic_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_local, sum_storage);
            if (v_means2d_abs != nullptr)
                rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_abs_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_opacity_local, sum_storage);

            int32_t g = warp.shfl(_id_batch, t);
            if (lane == 0) { // one atomicAdd set per gaussian per tile
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
        }
    }
}

// ---------------------------------------------------------------------------
// Wave32 (gfx1250) MULTI-TILE variant: a 256-thread block == 8 wave32 waves,
// where EACH wave independently rasterizes a DIFFERENT 8x8 tile using the exact
// bs32_1wave body (register+shfl geometry, 2 px/lane, per-wave rocprim (DPP)
// reduction, one atomicAdd set per Gaussian per tile). The 8 tiles handled by a
// block are (blockIdx.y * MT_WARPS + warp_id). Because the waves own disjoint
// tiles, there is NO cross-wave coupling: no block.sync(), no shared batch, and
// divergent per-wave loop trip counts / early returns are all safe. The point
// is to supply 8 resident waves per workgroup (deep occupancy, like tile16)
// while keeping the 8x8 tile granularity of bs32. Per-wave LDS (s_rgbs/s_vrc/
// s_buf) and rocprim scratch are replicated MT_WARPS times and indexed by
// warp_id. Gated to small CDIM in the launcher so 8x LDS fits in 64KB.
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
    const uint32_t tile_size,   // tile width in pixels
    const uint32_t tile_size_h, // tile height in pixels
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
    float T[2]; // H3: T_final/v_render_alpha moved to LDS (s_tf/s_va)
    int32_t bin_final_arr[2];

    // Per-wave dynamic LDS: MT_WARPS replicas of (s_rgbs | s_vrc | s_buf | s_tf |
    // s_va), selected by warp_id. Each wave only ever touches its own slab.
    // H3: s_tf (T_final) and s_va (v_render_alpha) are per-pixel and read-only
    // after setup; keeping them in LDS instead of registers trades VGPR for LDS.
    const size_t per_warp = (size_t)max_batch_size * CDIM      // s_rgbs
                          + (size_t)2 * GSPLAT_WARP_SIZE * CDIM // s_vrc
                          + (size_t)2 * GSPLAT_WARP_SIZE * CDIM // s_buf
                          + (size_t)2 * GSPLAT_WARP_SIZE        // s_tf
                          + (size_t)2 * GSPLAT_WARP_SIZE;       // s_va
    extern __shared__ float s_mem[];
    float *s_base = s_mem + (size_t)warp_id * per_warp;
    float *s_rgbs = s_base;
    float *s_vrc = s_base + (size_t)max_batch_size * CDIM;
    float *s_buf = s_vrc + (size_t)2 * GSPLAT_WARP_SIZE * CDIM;
    float *s_tf = s_buf + (size_t)2 * GSPLAT_WARP_SIZE * CDIM;
    float *s_va = s_tf + (size_t)2 * GSPLAT_WARP_SIZE;

#pragma unroll
    for (int s = 0; s < 2; ++s) {
        uint32_t p = lane + 32u * (uint32_t)s; // 0..63 within the tile
        uint32_t row = p / tile_size;
        uint32_t col = p % tile_size;
        uint32_t i = trow * tile_size_h + row;
        uint32_t j = tcol * tile_size + col;
        px[s] = (float)j + 0.5f;
        py[s] = (float)i + 0.5f;
        // H1: pix_id is dead after this loop -> loop-local, not a kernel-wide array.
        const int32_t pix_id =
            min(i * image_width + j, image_width * image_height - 1);
        const bool inside = (i < image_height && j < image_width);
        const float tf_h3 = 1.0f - render_alphas[pix_id];
        s_tf[p] = tf_h3; T[s] = tf_h3;
        // H2: outside pixels get bin_final = -1. The inner-loop test
        // (batch_end - t <= bin_final_arr[s]) is then always false for them,
        // so a separate kernel-wide inside_arr[2] is no longer needed.
        bin_final_arr[s] = inside ? last_ids[pix_id] : -1;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            s_buf[p * CDIM + k] = 0.f;
            s_vrc[p * CDIM + k] = v_render_colors[pix_id * CDIM + k];
        }
        s_va[p] = v_render_alphas[pix_id];
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
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            _xy_opacity_batch = {xy.x, xy.y, opac};
            _conic_batch = conics[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                s_rgbs[lane * CDIM + k] = colors[g * CDIM + k];
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

            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            bool any_valid = false;

#pragma unroll
            for (int s = 0; s < 2; ++s) {
                // H2: inside && (batch_end - t <= bin_final). Outside pixels have
                // bin_final_arr[s] == -1, so this is false for them (batch_end-t>=0).
                bool valid = (batch_end - t <= bin_final_arr[s]);
                float alpha;
                float opac_s;
                float vis;
                vec2 delta;
                if (valid) {
                    opac_s = xy_opac.z;
                    delta = {xy_opac.x - px[s], xy_opac.y - py[s]};
                    float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                          conic.z * delta.y * delta.y) +
                                  conic.y * delta.x * delta.y;
                    vis = __expf(-sigma);
                    alpha = min(0.999f, opac_s * vis);
                    if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                        valid = false;
                    }
                }
                if (valid) {
                    const uint32_t p_s = lane + GSPLAT_WARP_SIZE * (uint32_t)s;
                    const float *vrc_s = s_vrc + p_s * CDIM;
                    float *buf_s = s_buf + p_s * CDIM;
                    float ra = 1.0f / (1.0f - alpha);
                    T[s] *= ra;
                    const float fac = alpha * T[s];
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_rgb_local[k] += fac * vrc_s[k];
                    }
                    float v_alpha = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_alpha += (s_rgbs[t * CDIM + k] * T[s] - buf_s[k] * ra) *
                                   vrc_s[k];
                    }
                    v_alpha += s_tf[p_s] * ra * s_va[p_s];
                    if (backgrounds != nullptr) {
                        float accum = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            accum += backgrounds[k] * vrc_s[k];
                        }
                        v_alpha += -s_tf[p_s] * ra * accum;
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
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        buf_s[k] += s_rgbs[t * CDIM + k] * fac;
                    }
                }
                any_valid |= valid;
            }

            // skip the (relatively expensive) reduction if no pixel is active
            if (!warp.any(any_valid)) {
                continue;
            }

            // single 32-lane rocprim (DPP) reduction across this wave's tile
            rocprim_warpSum<CDIM, GSPLAT_WARP_SIZE>(v_rgb_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_conic_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_local, sum_storage);
            if constexpr (ABSGRAD)
                rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_abs_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_opacity_local, sum_storage);

            int32_t g = warp.shfl(_id_batch, t);
            if (lane == 0) { // one atomicAdd set per gaussian per tile
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

                if constexpr (ABSGRAD) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    atomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    atomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                atomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

// bs32_1to1 is DISABLED for now to avoid confusion: it only ran for the
// non-default block_size==32 (e.g. 8x4) tile config, and the corresponding
// launcher branches have been removed, so such tiles now fall back to the
// generic multi-warp kernel. Kept here (disabled) for reference / A-B.
// #if 0
// // ---------------------------------------------------------------------------
// // Wave32 (gfx1250) 1:1 variant: a SINGLE 32-lane wave covers a 32-pixel tile
// // (e.g. 8 wide x 4 tall) with EXACTLY ONE pixel per lane -- the true wave32
// // analog of the wave64 bs64 kernel. Per-Gaussian id/mean/opacity/conic live in
// // per-lane registers and are broadcast within the wave via warp.shfl(); only the
// // color batch (rgbs) is staged in dynamic LDS. One 32-lane rocprim (DPP)
// // reduction collapses each Gaussian's gradients across the 32 pixels and lane 0
// // issues a SINGLE atomicAdd set per Gaussian per tile (1 atomic/Gaussian/tile,
// // each Gaussian loaded once). Launched with a 2D {tile_size, tile_size_h} block
// // (== 32 threads), so pixel<->lane mapping is the same as the generic kernel.
// // Because there is only ONE pixel per lane, per-pixel register state (T, buffer,
// // v_render_c) is NOT doubled vs the 8x8 2-px/lane kernel.
// template <uint32_t CDIM, typename scalar_t>
// __launch_bounds__(32)
// __global__ void rasterize_bs32_1to1_to_pixels_3dgs_bwd_kernel(
//     const uint32_t I,
//     const uint32_t N,
//     const uint32_t n_isects,
//     const bool packed,
//     // fwd inputs
//     const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
//     const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
//     const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
//     const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
//     const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
//     const bool *__restrict__ masks,           // [..., tile_height, tile_width]
//     const uint32_t image_width,
//     const uint32_t image_height,
//     const uint32_t tile_size,   // tile width in pixels
//     const uint32_t tile_size_h, // tile height in pixels
//     const uint32_t tile_width,
//     const uint32_t tile_height,
//     const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
//     const int32_t *__restrict__ flatten_ids,  // [n_isects]
//     // fwd outputs
//     const scalar_t
//         *__restrict__ render_alphas,      // [..., image_height, image_width, 1]
//     const int32_t *__restrict__ last_ids, // [..., image_height, image_width]
//     // grad outputs
//     const scalar_t *__restrict__ v_render_colors, // [..., image_height,
//                                                   // image_width, CDIM]
//     const scalar_t
//         *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
//     // grad inputs
//     vec2 *__restrict__ v_means2d_abs,  // [..., N, 2] or [nnz, 2]
//     vec2 *__restrict__ v_means2d,      // [..., N, 2] or [nnz, 2]
//     vec3 *__restrict__ v_conics,       // [..., N, 3] or [nnz, 3]
//     scalar_t *__restrict__ v_colors,   // [..., N, CDIM] or [nnz, CDIM]
//     scalar_t *__restrict__ v_opacities, // [..., N] or [nnz]
//     const uint32_t max_batch_size
// ) {
//     auto block = cg::this_thread_block();
//     uint32_t image_id = block.group_index().x;
//     uint32_t tile_id =
//         block.group_index().y * tile_width + block.group_index().z;
//     uint32_t i = block.group_index().y * tile_size_h + block.thread_index().y;
//     uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

//     tile_offsets += image_id * tile_height * tile_width;
//     render_alphas += image_id * image_height * image_width;
//     last_ids += image_id * image_height * image_width;
//     v_render_colors += image_id * image_height * image_width * CDIM;
//     v_render_alphas += image_id * image_height * image_width;
//     if (backgrounds != nullptr) {
//         backgrounds += image_id * CDIM;
//     }
//     if (masks != nullptr) {
//         masks += image_id * tile_height * tile_width;
//     }

//     // when the mask is provided, do nothing and return if
//     // this tile is labeled as False
//     if (masks != nullptr && !masks[tile_id]) {
//         return;
//     }

//     const float px = (float)j + 0.5f;
//     const float py = (float)i + 0.5f;
//     // clamp this value to the last pixel
//     const int32_t pix_id =
//         min(i * image_width + j, image_width * image_height - 1);
//     // keep not rasterizing threads around for reading data
//     bool inside = (i < image_height && j < image_width);

//     // which gaussians to look through in this tile
//     int32_t range_start = tile_offsets[tile_id];
//     int32_t range_end =
//         (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
//             ? n_isects
//             : tile_offsets[tile_id + 1];

//     const uint32_t num_batches =
//         (range_end - range_start + max_batch_size - 1) / max_batch_size;

//     // dynamic LDS: color batch [max_batch_size * CDIM] followed by one rocprim
//     // warp_reduce storage object (single 32-lane wave -> single warp).
//     extern __shared__ float s_bs32_1to1[];
//     float *rgbs_batch = s_bs32_1to1; // [max_batch_size * CDIM]
//     using warp_reduce_float_t = rocprim::warp_reduce<float, GSPLAT_WARP_SIZE>;
//     auto *warp_storage_base = (typename warp_reduce_float_t::storage_type *)(
//         rgbs_batch + (size_t)max_batch_size * CDIM);

//     // this is the T AFTER the last gaussian in this pixel
//     float T_final = 1.0f - render_alphas[pix_id];
//     float T = T_final;
//     // the contribution from gaussians behind the current one
//     float buffer[CDIM] = {0.f};
//     // index of last gaussian to contribute to this pixel
//     const int32_t bin_final = inside ? last_ids[pix_id] : 0;

//     // df/d_out for this pixel
//     float v_render_c[CDIM];
// #pragma unroll
//     for (uint32_t k = 0; k < CDIM; ++k) {
//         v_render_c[k] = v_render_colors[pix_id * CDIM + k];
//     }
//     const float v_render_a = v_render_alphas[pix_id];

//     cg::thread_block_tile<GSPLAT_WARP_SIZE> warp =
//         cg::tiled_partition<GSPLAT_WARP_SIZE>(block);
//     const uint32_t lane = warp.thread_rank();

//     __shared__ typename rocprim::warp_reduce<int32_t, GSPLAT_WARP_SIZE>::storage_type
//         bin_storage[1];
//     rocprim::warp_reduce<int32_t, GSPLAT_WARP_SIZE> bin_reduce;
//     int32_t warp_bin_final;
//     bin_reduce.reduce(
//         bin_final, warp_bin_final, bin_storage[0], rocprim::maximum<int32_t>());

//     int32_t _id_batch;
//     vec3 _xy_opacity_batch;
//     vec3 _conic_batch;
//     for (uint32_t b = 0; b < num_batches; ++b) {
//         warp.sync();

//         // one wave of 32 lanes loads 32 gaussians (each once, no duplication)
//         const int64_t batch_end = range_end - 1 - max_batch_size * b;
//         const uint32_t current_batch_size =
//             (uint32_t)min((int64_t)max_batch_size, batch_end + 1 - range_start);
//         const int64_t idx = batch_end - lane;
//         if (lane < current_batch_size && idx >= range_start) {
//             int32_t g = flatten_ids[idx];
//             _id_batch = g;
//             const vec2 xy = means2d[g];
//             const float opac = opacities[g];
//             _xy_opacity_batch = {xy.x, xy.y, opac};
//             _conic_batch = conics[g];
// #pragma unroll
//             for (uint32_t k = 0; k < CDIM; ++k) {
//                 rgbs_batch[lane * CDIM + k] = colors[g * CDIM + k];
//             }
//         }
//         warp.sync();

//         for (uint32_t t = max(0, batch_end - warp_bin_final);
//              t < current_batch_size; ++t) {
//             bool valid = inside;
//             if (batch_end - t > bin_final) {
//                 valid = false;
//             }
//             // broadcast gaussian t (held by lane t) to every lane in the wave
//             vec3 conic;
//             conic.x = warp.shfl(_conic_batch.x, t);
//             conic.y = warp.shfl(_conic_batch.y, t);
//             conic.z = warp.shfl(_conic_batch.z, t);
//             vec3 xy_opac;
//             xy_opac.x = warp.shfl(_xy_opacity_batch.x, t);
//             xy_opac.y = warp.shfl(_xy_opacity_batch.y, t);
//             xy_opac.z = warp.shfl(_xy_opacity_batch.z, t);
//             float alpha;
//             float opac;
//             vec2 delta;
//             float vis;
//             if (valid) {
//                 opac = xy_opac.z;
//                 delta = {xy_opac.x - px, xy_opac.y - py};
//                 float sigma = 0.5f * (conic.x * delta.x * delta.x +
//                                       conic.z * delta.y * delta.y) +
//                               conic.y * delta.x * delta.y;
//                 vis = __expf(-sigma);
//                 alpha = min(0.999f, opac * vis);
//                 if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
//                     valid = false;
//                 }
//             }

//             // if all threads are inactive in this warp, skip this loop
//             if (!warp.any(valid)) {
//                 continue;
//             }
//             float v_rgb_local[CDIM] = {0.f};
//             vec3 v_conic_local = {0.f, 0.f, 0.f};
//             vec2 v_xy_local = {0.f, 0.f};
//             vec2 v_xy_abs_local = {0.f, 0.f};
//             float v_opacity_local = 0.f;
//             if (valid) {
//                 float ra = 1.0f / (1.0f - alpha);
//                 T *= ra;
//                 const float fac = alpha * T;
// #pragma unroll
//                 for (uint32_t k = 0; k < CDIM; ++k) {
//                     v_rgb_local[k] = fac * v_render_c[k];
//                 }
//                 float v_alpha = 0.f;
// #pragma unroll
//                 for (uint32_t k = 0; k < CDIM; ++k) {
//                     v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
//                                v_render_c[k];
//                 }
//                 v_alpha += T_final * ra * v_render_a;
//                 if (backgrounds != nullptr) {
//                     float accum = 0.f;
// #pragma unroll
//                     for (uint32_t k = 0; k < CDIM; ++k) {
//                         accum += backgrounds[k] * v_render_c[k];
//                     }
//                     v_alpha += -T_final * ra * accum;
//                 }
//                 if (opac * vis <= 0.999f) {
//                     const float v_sigma = -opac * vis * v_alpha;
//                     v_conic_local = {
//                         0.5f * v_sigma * delta.x * delta.x,
//                         v_sigma * delta.x * delta.y,
//                         0.5f * v_sigma * delta.y * delta.y
//                     };
//                     v_xy_local = {
//                         v_sigma * (conic.x * delta.x + conic.y * delta.y),
//                         v_sigma * (conic.y * delta.x + conic.z * delta.y)
//                     };
//                     if (v_means2d_abs != nullptr) {
//                         v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
//                     }
//                     v_opacity_local = vis * v_alpha;
//                 }
// #pragma unroll
//                 for (uint32_t k = 0; k < CDIM; ++k) {
//                     buffer[k] += rgbs_batch[t * CDIM + k] * fac;
//                 }
//             }

//             // single 32-lane rocprim (DPP) reduction across the whole tile
//             rocprim_warpSum<CDIM, GSPLAT_WARP_SIZE>(v_rgb_local, warp_storage_base);
//             rocprim_warpSum<GSPLAT_WARP_SIZE>(v_conic_local, warp_storage_base);
//             rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_local, warp_storage_base);
//             if (v_means2d_abs != nullptr)
//                 rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_abs_local, warp_storage_base);
//             rocprim_warpSum<GSPLAT_WARP_SIZE>(v_opacity_local, warp_storage_base);

//             int32_t g = warp.shfl(_id_batch, t);
//             if (lane == 0) { // one atomicAdd set per gaussian per tile
//                 float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
// #pragma unroll
//                 for (uint32_t k = 0; k < CDIM; ++k) {
//                     atomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
//                 }

//                 float *v_conic_ptr = (float *)(v_conics) + 3 * g;
//                 atomicAdd(v_conic_ptr, v_conic_local.x);
//                 atomicAdd(v_conic_ptr + 1, v_conic_local.y);
//                 atomicAdd(v_conic_ptr + 2, v_conic_local.z);

//                 float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
//                 atomicAdd(v_xy_ptr, v_xy_local.x);
//                 atomicAdd(v_xy_ptr + 1, v_xy_local.y);

//                 if (v_means2d_abs != nullptr) {
//                     float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
//                     atomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
//                     atomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
//                 }

//                 atomicAdd(v_opacities + g, v_opacity_local);
//             }
//         }
//     }
// }
// #endif // bs32_1to1 disabled

// ---------------------------------------------------------------------------
// Wave32 (gfx1250) 2-wave cooperative variant: a full 8x8 (64-pixel) tile is
// covered by a 64-thread block == TWO 32-lane waves, ONE pixel per lane. This
// keeps the wave64 "bs64" advantages on a wave32 device while avoiding the
// bs32_1wave pitfalls:
//   * 1 px/lane  -> per-pixel state (T, buffer[CDIM], v_render_c[CDIM]) is NOT
//                   doubled and lives in registers (no LDS in the hot loop).
//   * 8x8 tile   -> same intersection count / footprint coverage as bs64 (no
//                   tile-count blow-up like the 8x4 bs32_1to1 kernel).
//   * cross-wave -> each wave rocprim-reduces its own 32 pixels; the two per-
//                   wave partials are combined through a tiny static-LDS slot so
//                   lane 0 of wave 0 issues a SINGLE atomicAdd set per Gaussian
//                   per tile (1 atomic/Gaussian/tile, matching bs64 -- NOT the
//                   2 atomics/Gaussian/tile of the naive 2-wave kernel).
// The Gaussian batch (id/mean/opacity/conic/colors) is staged in dynamic LDS and
// read by both waves (cross-wave broadcast cannot use shfl). Correctness across
// the two waves requires block-uniform control flow: bin_final is reduced over
// the WHOLE block (so both waves iterate the same Gaussian range) and the
// per-Gaussian "all inactive" skip uses __syncthreads_or (a block-wide barrier)
// so the two waves never diverge before a block.sync().
template <uint32_t CDIM, typename scalar_t>
__launch_bounds__(64)
__global__ void rasterize_bs32_2wave_coop_to_pixels_3dgs_bwd_kernel(
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
    const uint32_t tile_size_h, // tile height in pixels
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

    // 1D block of 64 threads == 2 wave32 waves. Map the linear thread rank to a
    // pixel of the tile: p = 0..63, row = p / tile_size, col = p % tile_size.
    // Using the linear rank (not a 2D block) keeps threadIdx.x == thread_rank,
    // which is what rocprim_warpSum uses to index its per-warp LDS storage
    // (warp_id = threadIdx.x / 32).
    const uint32_t p = block.thread_rank();     // 0..63 within the tile
    const uint32_t row = p / tile_size;
    const uint32_t col = p % tile_size;
    uint32_t i = block.group_index().y * tile_size_h + row;
    uint32_t j = block.group_index().z * tile_size + col;

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
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);
    bool inside = (i < image_height && j < image_width);

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];

    const uint32_t num_batches =
        (range_end - range_start + max_batch_size - 1) / max_batch_size;

    // Dynamic LDS Gaussian batch (staged once, read by BOTH waves):
    //   id_batch [B] | xy_opacity_batch [B] | conic_batch [B] | rgbs_batch [B*CDIM]
    extern __shared__ int s_coop[];
    int32_t *id_batch = (int32_t *)s_coop;
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[max_batch_size]);
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[max_batch_size]);
    float *rgbs_batch = (float *)&conic_batch[max_batch_size];

    // per-pixel state (single copy, in registers) -- 1 pixel per lane
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    float buffer[CDIM] = {0.f};
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<GSPLAT_WARP_SIZE> warp =
        cg::tiled_partition<GSPLAT_WARP_SIZE>(block);
    const uint32_t lane = warp.thread_rank();          // 0..31
    const uint32_t warp_id = tr / GSPLAT_WARP_SIZE;    // 0 or 1

    // Per-warp rocprim reduction scratch (one storage object per wave).
    __shared__ typename rocprim::warp_reduce<float, GSPLAT_WARP_SIZE>::storage_type
        sum_storage[2];

    // Cross-wave combine slot: wave 1's per-Gaussian partials, read by wave 0.
    __shared__ float cmb_rgb[CDIM];
    __shared__ float cmb_conic[3];
    __shared__ float cmb_xy[2];
    __shared__ float cmb_xy_abs[2];
    __shared__ float cmb_opac;

    // block-wide (all 64 pixels) max of bin_final so BOTH waves iterate the same
    // Gaussian range -> every thread reaches the block.sync() calls below.
    __shared__ int32_t s_block_bin_final;
    if (tr == 0) {
        s_block_bin_final = 0;
    }
    block.sync();
    atomicMax(&s_block_bin_final, bin_final);
    block.sync();
    const int32_t warp_bin_final = s_block_bin_final;

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        // 64 threads load 64 gaussians (each once) from back to front.
        const int64_t batch_end = range_end - 1 - max_batch_size * b;
        const uint32_t current_batch_size =
            (uint32_t)min((int64_t)max_batch_size, batch_end + 1 - range_start);
        const int64_t idx = batch_end - tr;
        if (tr < current_batch_size && idx >= range_start) {
            int32_t g = flatten_ids[idx];
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
        block.sync();

        for (uint32_t t = max(0, batch_end - warp_bin_final);
             t < current_batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = false;
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

            // Block-wide "any active?" test that ALSO acts as a barrier. Using a
            // block-wide (not per-wave) predicate keeps the two waves in lockstep
            // so neither diverges before the block.sync() further below.
            if (!__syncthreads_or(valid)) {
                continue;
            }

            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            if (valid) {
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }
                v_alpha += T_final * ra * v_render_a;
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

            // Each wave reduces across its own 32 pixels (result valid in lane 0
            // of the wave). sum_storage is indexed per-warp inside rocprim.
            rocprim_warpSum<CDIM, GSPLAT_WARP_SIZE>(v_rgb_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_conic_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_local, sum_storage);
            if (v_means2d_abs != nullptr)
                rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_abs_local, sum_storage);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_opacity_local, sum_storage);

            // wave 1 publishes its partial to LDS; wave 0 reads it after the
            // barrier and adds its own (register) partial -> ONE atomic set.
            if (warp_id == 1 && lane == 0) {
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    cmb_rgb[k] = v_rgb_local[k];
                }
                cmb_conic[0] = v_conic_local.x;
                cmb_conic[1] = v_conic_local.y;
                cmb_conic[2] = v_conic_local.z;
                cmb_xy[0] = v_xy_local.x;
                cmb_xy[1] = v_xy_local.y;
                if (v_means2d_abs != nullptr) {
                    cmb_xy_abs[0] = v_xy_abs_local.x;
                    cmb_xy_abs[1] = v_xy_abs_local.y;
                }
                cmb_opac = v_opacity_local;
            }
            block.sync();

            if (warp_id == 0 && lane == 0) {
                int32_t g = id_batch[t];
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    atomicAdd(v_rgb_ptr + k, v_rgb_local[k] + cmb_rgb[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                atomicAdd(v_conic_ptr, v_conic_local.x + cmb_conic[0]);
                atomicAdd(v_conic_ptr + 1, v_conic_local.y + cmb_conic[1]);
                atomicAdd(v_conic_ptr + 2, v_conic_local.z + cmb_conic[2]);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                atomicAdd(v_xy_ptr, v_xy_local.x + cmb_xy[0]);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y + cmb_xy[1]);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    atomicAdd(v_xy_abs_ptr, v_xy_abs_local.x + cmb_xy_abs[0]);
                    atomicAdd(
                        v_xy_abs_ptr + 1, v_xy_abs_local.y + cmb_xy_abs[1]
                    );
                }

                atomicAdd(v_opacities + g, v_opacity_local + cmb_opac);
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
    const uint32_t tile_size_h, // tile height in pixels
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
    uint32_t i = block.group_index().y * tile_size_h + block.thread_index().y;
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
    const uint32_t tile_size_h, // tile height in pixels
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
    dim3 threads = {tile_size, tile_size_h, 1};
    dim3 grid = {I, tile_height, tile_width};

#if USE_ROCM
    // Optimization for ROCm: Use smaller batch size to reduce shared memory usage

    const uint32_t block_size = tile_size * tile_size_h;
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
    // wave32 register+shfl path (1-wave): an 8x8 tile is handled by a SINGLE
    // 32-lane wave with 2 pixels per lane. means2d/opacity/conic live in
    // registers (broadcast via shfl); the color batch (s_rgbs) and per-pixel
    // v_render_colors gradient (s_vrc) are staged in dynamic LDS to reduce VGPR
    // pressure. The block is launched with 32 threads. Gated to small CDIM so
    // the doubled per-pixel register state does not spill.
    if (block_size == 64 && CDIM <= 32) {
#if GSPLAT_BS32_MULTITILE
      if constexpr (CDIM <= GSPLAT_BS32_MULTITILE_MAXCDIM) {
      // multi-tile path: a 256-thread block (MT_WARPS == 8 wave32 waves) where
      // each wave independently rasterizes ONE 8x8 tile (bs32_1wave body). The
      // per-wave LDS (s_rgbs|s_vrc|s_buf) is replicated MT_WARPS times; the grid
      // packs MT_WARPS tiles per block. Gated to small CDIM so 8x LDS fits 64KB.
      constexpr uint32_t MT_WARPS = 256u / GSPLAT_WARP_SIZE; // 8
      max_batch_size = GSPLAT_WARP_SIZE; // 32, each wave loads its own batch
      shmem_size = (int64_t)MT_WARPS *
        ((int64_t)GSPLAT_WARP_SIZE * CDIM * sizeof(float)         // s_rgbs
         + (int64_t)2 * GSPLAT_WARP_SIZE * CDIM * sizeof(float)   // s_vrc
         + (int64_t)2 * GSPLAT_WARP_SIZE * CDIM * sizeof(float)   // s_buf
         + (int64_t)2 * GSPLAT_WARP_SIZE * sizeof(float)          // s_tf  (H3)
         + (int64_t)2 * GSPLAT_WARP_SIZE * sizeof(float));        // s_va  (H3)
      threads = dim3{256, 1, 1};
      grid = dim3{I, (tile_height * tile_width + MT_WARPS - 1) / MT_WARPS, 1};
      } else
#endif
      {
#if GSPLAT_BS32_2WAVE_COOP
      // 2-wave cooperative path: a 64-thread block (== two 32-lane waves) covers
      // the 8x8 tile, 1 pixel per lane. The Gaussian batch (id/mean/opacity/
      // conic/colors) is staged in dynamic LDS and read by both waves; the
      // rocprim reduction scratch and the cross-wave combine slot are static
      // shared, so they stay out of this dynamic size. 64 threads load 64
      // gaussians per batch (one each).
      max_batch_size = block_size; // 64
      shmem_size = (int64_t)max_batch_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);
      threads = dim3{64, 1, 1}; // 1D so threadIdx.x == thread_rank (0..63)
#else
      max_batch_size = GSPLAT_WARP_SIZE; // 32
      // LDS: colors batch [max_batch_size*CDIM] + per-pixel v_render_c
      // [2*GSPLAT_WARP_SIZE*CDIM] + per-pixel accumulator [2*GSPLAT_WARP_SIZE*CDIM]
      // (64 pixels/tile).
      shmem_size = (int64_t)GSPLAT_WARP_SIZE * CDIM * sizeof(float)       // s_rgbs
                 + (int64_t)2 * GSPLAT_WARP_SIZE * CDIM * sizeof(float)   // s_vrc
                 + (int64_t)2 * GSPLAT_WARP_SIZE * CDIM * sizeof(float);  // s_buf
      threads = dim3{GSPLAT_WARP_SIZE, 1, 1}; // one wave, 2 px/thread
#endif
      }
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
    // Wave32: default to the generic multi-warp kernel. For the square 8x8
    // (block_size==64) tile with small CDIM, use a specialized single-block
    // kernel: the 2-wave cooperative kernel (GSPLAT_BS32_2WAVE_COOP=1, default)
    // or the 1-wave / 2-px-per-lane kernel. All other tile sizes (incl. 32)
    // fall back to the generic kernel.
    if constexpr (CDIM <= 32) {
        if (block_size == 64) {
#if GSPLAT_BS32_MULTITILE
            if constexpr (CDIM <= GSPLAT_BS32_MULTITILE_MAXCDIM) {
                // H8: pick the ABSGRAD instantiation so the abs path is compiled
                // out when the caller does not request abs gradients.
                KERNEL = v_means2d_abs.has_value()
                    ? rasterize_bs32_8tile_to_pixels_3dgs_bwd_kernel<CDIM, float, true>
                    : rasterize_bs32_8tile_to_pixels_3dgs_bwd_kernel<CDIM, float, false>;
            } else
#endif
            {
#if GSPLAT_BS32_2WAVE_COOP
            KERNEL =
                rasterize_bs32_2wave_coop_to_pixels_3dgs_bwd_kernel<CDIM, float>;
#else
            KERNEL = rasterize_bs32_1wave_to_pixels_3dgs_bwd_kernel<CDIM, float>;
#endif
            }
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
            tile_size_h,
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
        uint32_t tile_size_h,                                                  \
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

