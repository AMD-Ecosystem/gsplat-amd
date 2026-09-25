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

//compiler issue with mov_dpp intrinsic seen in Rocm 6.4.1, so mov_dpp intrinsic is temporarily commented out and replaced with rocprim which also uses dpp when in single wave
#if USE_ROCM
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
    const uint32_t tile_size,
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
    const uint32_t tile_size,
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
    using warp_reduce_float_t = rocprim::warp_reduce<float,64>;
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
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);
    #else
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    #endif
    
    #if USE_ROCM
        __shared__ typename rocprim::warp_reduce<int32_t, 64>::storage_type warp_storage;
        rocprim::warp_reduce<int32_t, 64> wreduce;
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
            rocprim_warpSum<CDIM, 64>(v_rgb_local, warp_storage_base);   // CDIM-sized float array
            rocprim_warpSum<64>(v_conic_local, warp_storage_base); // float
            rocprim_warpSum<64>(v_xy_local, warp_storage_base);    // vec2
            if (v_means2d_abs != nullptr)
                rocprim_warpSum<64>(v_xy_abs_local, warp_storage_base);// vec2
            rocprim_warpSum<64>(v_opacity_local, warp_storage_base);// float
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
    const uint32_t tile_size,
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
    if (block_size == 64) { // wave64-optimized path
      max_batch_size = 32;
      //max_batch_size = min(max_batch_size, block_size);
      if (CDIM <= 32) {
        max_batch_size = block_size;
      }
      shmem_size =
        max_batch_size *
        (sizeof(float) * CDIM);
    } else {
      max_batch_size = 16;
      max_batch_size = min(max_batch_size, block_size);
      if (CDIM <= 16) {
        max_batch_size = block_size;
      }
      const uint32_t warps_per_block = (block_size + 63) / 64; // for 64-lane warp
      std::size_t warp_scratch_bytes =
        warps_per_block * sizeof(typename rocprim::warp_reduce<float,64>::storage_type);
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
    auto KERNEL = (block_size == 64) ?
	    rasterize_bs64_to_pixels_3dgs_bwd_kernel<CDIM, float> :
	    rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>;
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

