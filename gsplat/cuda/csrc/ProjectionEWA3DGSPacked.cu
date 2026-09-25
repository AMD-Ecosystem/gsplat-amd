#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

#include "Common.h"
#include "Common.cuh"
#include "Projection.h"
#include "Utils.cuh"

#define DEBUG_PRINT 0
#ifdef DEBUG_PRINT
#include <cstdio> // Only include cstdio if DEBUG_PRINT is enabled
#endif

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void projection_ewa_3dgs_packed_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ covars,   // [B, N, 6] Optional
    const scalar_t *__restrict__ quats,    // [B, N, 4] Optional
    const scalar_t *__restrict__ scales,   // [B, N, 3] Optional
    const scalar_t *__restrict__ opacities, // [B, N] optional
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const int32_t
        *__restrict__ block_accum, // [B * C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ block_cnts,    // [B * C * blocks_per_row] packing helper
    int32_t *__restrict__ indptr,        // [B * C + 1]
    int64_t *__restrict__ batch_ids,       // [nnz]
    int64_t *__restrict__ camera_ids,    // [nnz]
    int64_t *__restrict__ gaussian_ids,  // [nnz]
    int32_t *__restrict__ radii,         // [nnz, 2]
    scalar_t *__restrict__ means2d,      // [nnz, 2]
    scalar_t *__restrict__ depths,       // [nnz]
    scalar_t *__restrict__ conics,       // [nnz, 3]
    scalar_t *__restrict__ compensations // [nnz] optional
) {
    int32_t blocks_per_row = gridDim.x;
    int32_t row_idx = blockIdx.y;
    int32_t block_col_idx = blockIdx.x;
    int32_t block_idx = row_idx * blocks_per_row + block_col_idx;
    int32_t col_idx = block_col_idx * blockDim.x + threadIdx.x;
    const int32_t bid = row_idx / C;
    const int32_t cid = row_idx % C;
    const int32_t gid = col_idx;

    bool valid = (bid < B) && (cid < C) && (gid < N);

    // check if points are with camera near and far plane
    vec3 mean_c;
    mat3 R;
    if (valid) {
        // shift pointers to the current camera and gaussian
        means += bid * N * 3 + gid * 3;
        viewmats += bid * C * 16 + cid * 16;

        // glm is column-major but input is row-major
        R = mat3(
            viewmats[0],
            viewmats[4],
            viewmats[8], // 1st column
            viewmats[1],
            viewmats[5],
            viewmats[9], // 2nd column
            viewmats[2],
            viewmats[6],
            viewmats[10] // 3rd column
        );
        vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

        // transform Gaussian center to camera space
        posW2C(R, t, glm::make_vec3(means), mean_c);
        if (mean_c.z < near_plane || mean_c.z > far_plane) {
            valid = false;
        }
    }

    // check if the perspective projection is valid.
    mat2 covar2d;
    vec2 mean2d;
    mat2 covar2d_inv;
    float compensation;
    float det;
    if (valid) {
        // transform Gaussian covariance to camera space
        mat3 covar;
        if (covars != nullptr) {
            // if a precomputed covariance is provided
            covars += bid * N * 6 + gid * 6;
            covar = mat3(
                covars[0],
                covars[1],
                covars[2], // 1st column
                covars[1],
                covars[3],
                covars[4], // 2nd column
                covars[2],
                covars[4],
                covars[5] // 3rd column
            );
        } else {
            // if not then compute it from quaternions and scales
            quats += bid * N * 4 + gid * 4;
            scales += bid * N * 3 + gid * 3;
            quat_scale_to_covar_preci(
                glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
            );
        }
        mat3 covar_c;
        covarW2C(R, covar, covar_c);

        Ks += bid * C * 9 + cid * 9;
        switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
            #if USE_ROCM
                near_plane,
            #endif
                covar2d,
                mean2d
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
        }

        det = add_blur(eps2d, covar2d, compensation);
        if (det <= 0.f) {
            valid = false;
        } else {
            // compute the inverse of the 2d covariance
            covar2d_inv = glm::inverse(covar2d);
        }
    }

    // check if the points are in the image region
    float radius_x, radius_y;
    if (valid) {
        float extend = 3.33f;
        if (opacities != nullptr) {
            float opacity = opacities[col_idx];
            if (compensations != nullptr) {
                // we assume compensation term will be applied later on.
                opacity *= compensation;
            }    
            if (opacity < ALPHA_THRESHOLD) {
                valid = false;
            }
            // Compute opacity-aware bounding box.
            // https://arxiv.org/pdf/2402.00525 Section B.2
            extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
        }
        
        // compute tight rectangular bounding box (non differentiable)
        // https://arxiv.org/pdf/2402.00525
        radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
        radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

        if (radius_x <= radius_clip && radius_y <= radius_clip) {
            valid = false;
        }

        // mask out gaussians outside the image region
        if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
            mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
            valid = false;
        }
    }

    int32_t thread_data = static_cast<int32_t>(valid);
    if (block_cnts != nullptr) {
        // First pass: compute the block-wide sum
        int32_t aggregate;
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockReduce<int32_t, N_THREADS_PACKED> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
        } else {
            aggregate = 0;
        }
        if (threadIdx.x == 0) {
            block_cnts[block_idx] = aggregate;
        }
    } else {
        // Second pass: write out the indices of the non zero elements
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockScan<int32_t, N_THREADS_PACKED> BlockScan;
            __shared__ typename BlockScan::TempStorage temp_storage;
            BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
        }
        if (valid) {
            if (block_idx > 0) {
                int32_t offset = block_accum[block_idx - 1];
                thread_data += offset;
            }
            // write to outputs
            batch_ids[thread_data] = bid;
            camera_ids[thread_data] = cid;
            gaussian_ids[thread_data] = gid;
            radii[thread_data * 2] = (int32_t)radius_x;
            radii[thread_data * 2 + 1] = (int32_t)radius_y;
            means2d[thread_data * 2] = mean2d.x;
            means2d[thread_data * 2 + 1] = mean2d.y;
            depths[thread_data] = mean_c.z;
            conics[thread_data * 3] = covar2d_inv[0][0];
            conics[thread_data * 3 + 1] = covar2d_inv[0][1];
            conics[thread_data * 3 + 2] = covar2d_inv[1][1];
            if (compensations != nullptr) {
                compensations[thread_data] = compensation;
            }
        }
        // lane 0 of the first block in each row writes the indptr
        if (threadIdx.x == 0 && block_col_idx == 0) {
            if (row_idx == 0) {
                indptr[0] = 0;
                indptr[B * C] = block_accum[B * C * blocks_per_row - 1];
            } else {
                indptr[row_idx] = block_accum[block_idx - 1];
            }
        }
    }
}

void launch_projection_ewa_3dgs_packed_fwd_kernel(
    // inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const at::optional<at::Tensor>
        block_accum, // [B * C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    // outputs
    at::optional<at::Tensor> block_cnts,   // [B * C * blocks_per_row] packing helper
    at::optional<at::Tensor> indptr,       // [B * C + 1]
    at::optional<at::Tensor> batch_ids,    // [nnz]
    at::optional<at::Tensor> camera_ids,   // [nnz]
    at::optional<at::Tensor> gaussian_ids, // [nnz]
    at::optional<at::Tensor> radii,        // [nnz, 2]
    at::optional<at::Tensor> means2d,      // [nnz, 2]
    at::optional<at::Tensor> depths,       // [nnz]
    at::optional<at::Tensor> conics,       // [nnz, 3]
    at::optional<at::Tensor> compensations // [nnz] optional
) {
    uint32_t N = means.size(-2);          // number of gaussians
    uint32_t C = viewmats.size(-3);       // number of cameras
    uint32_t B = means.numel() / (N * 3); // number of batches

    uint32_t nrows = B * C;
    uint32_t ncols = N;
    uint32_t blocks_per_row = (ncols + N_THREADS_PACKED - 1) / N_THREADS_PACKED;

    dim3 threads(N_THREADS_PACKED);
    // limit on the number of blocks: [2**31 - 1, 65535, 65535]
    dim3 grid(blocks_per_row, nrows, 1);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (B == 0 || N == 0 || C == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_packed_fwd_kernel",
        [&]() {
            projection_ewa_3dgs_packed_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   GET_CURRENT_STREAM()>>>(
                    B,
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    quats.has_value() ? quats.value().data_ptr<scalar_t>()
                                      : nullptr,
                    scales.has_value() ? scales.value().data_ptr<scalar_t>()
                                       : nullptr,
                    opacities.has_value() ? opacities.value().data_ptr<scalar_t>()
                                       : nullptr,
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    block_accum.has_value()
                        ? block_accum.value().data_ptr<int32_t>()
                        : nullptr,
                    camera_model,
                    block_cnts.has_value()
                        ? block_cnts.value().data_ptr<int32_t>()
                        : nullptr,
                    indptr.has_value() ? indptr.value().data_ptr<int32_t>()
                                       : nullptr,
                    batch_ids.has_value()
                        ? batch_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    camera_ids.has_value()
                        ? camera_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    gaussian_ids.has_value()
                        ? gaussian_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    radii.has_value() ? radii.value().data_ptr<int32_t>()
                                      : nullptr,
                    means2d.has_value() ? means2d.value().data_ptr<scalar_t>()
                                        : nullptr,
                    depths.has_value() ? depths.value().data_ptr<scalar_t>()
                                       : nullptr,
                    conics.has_value() ? conics.value().data_ptr<scalar_t>()
                                       : nullptr,
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr
                );
        }
    );
}

// --- Main Kernel Function ---
template <typename scalar_t> // scalar_t for array pointers (can be float or double)
__global__ void projection_ewa_3dgs_packed_bwd_kernel(
    // fwd inputs
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const scalar_t *__restrict__ means,     // [B, N, 3]
    const scalar_t *__restrict__ covars,    // [B, N, 6] Optional
    const scalar_t *__restrict__ quats,     // [B, N, 4] Optional
    const scalar_t *__restrict__ scales,    // [B, N, 3] Optional
    const scalar_t *__restrict__ viewmats,  // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,        // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const scalar_t eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int64_t *__restrict__ batch_ids,       // [nnz]
    const int64_t *__restrict__ camera_ids,      // [nnz]
    const int64_t *__restrict__ gaussian_ids,    // [nnz]
    const scalar_t *__restrict__ conics,        // [nnz, 3]
    const scalar_t *__restrict__ compensations, // [nnz] optional
    // grad outputs (from downstream loss)
    const scalar_t *__restrict__ v_means2d,       // [nnz, 2]
    const scalar_t *__restrict__ v_depths,        // [nnz]
    const scalar_t *__restrict__ v_conics,        // [nnz, 3]
    const scalar_t *__restrict__ v_compensations, // [nnz] optional
    const bool sparse_grad, // whether the outputs are in COO format [nnz, ...]
    // grad inputs (accumulated gradients for parameters)
    scalar_t *__restrict__ v_means,    // [B, N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_covars,   // [B, N, 6] or [nnz, 6] Optional
    scalar_t *__restrict__ v_quats,    // [B, N, 4] or [nnz, 4] Optional
    scalar_t *__restrict__ v_scales,   // [B, N, 3] or [nnz, 3] Optional
    scalar_t *__restrict__ v_viewmats // [B, C, 4, 4] Optional
) {
    // Each thread processes one non-zero (nnz) entry.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= nnz) {
        return;
    }

    const int64_t bid = batch_ids[idx];    // batch id
    const int64_t cid = camera_ids[idx];   // camera id
    const int64_t gid = gaussian_ids[idx]; // gaussian id
    if (idx % 100000 == 0 && DEBUG_PRINT) {
        printf("\n--- Start Debug for idx = %u (Thread %u) ---\n", idx, threadIdx.x);
        printf("  B=%u, C=%u, N=%u, nnz=%u\n", B, C, N, nnz);
        printf("  bid=%ld, cid=%ld, gid=%ld\n", bid, cid, gid);
        printf("  sparse_grad=%d\n", (int)sparse_grad);

        // Print initial input pointers' values (cast to float for printf)
        printf("  means_in: %f, %f, %f\n", (float)means[bid*N*3 + gid*3 + 0], (float)means[bid*N*3 + gid*3 + 1], (float)means[bid*N*3 + gid*3 + 2]);
        if (covars) printf("  covars_in (0): %f\n", (float)covars[bid*N*6 + gid*6 + 0]);
        if (quats) printf("  quats_in: %f, %f, %f, %f\n", (float)quats[bid*N*4 + gid*4 + 0], (float)quats[bid*N*4 + gid*4 + 1], (float)quats[bid*N*4 + gid*4 + 2], (float)quats[bid*N*4 + gid*4 + 3]);
        if (scales) printf("  scales_in: %f, %f, %f\n", (float)scales[bid*N*3 + gid*3 + 0], (float)scales[bid*N*3 + gid*3 + 1], (float)scales[bid*N*3 + gid*3 + 2]);
        printf("  v_means2d_in: %f, %f\n", (float)v_means2d[idx*2+0], (float)v_means2d[idx*2+1]);
        printf("  v_depths_in: %f\n", (float)v_depths[idx]);
        printf("  v_conics_in: %f, %f, %f\n", (float)v_conics[idx*3+0], (float)v_conics[idx*3+1], (float)v_conics[idx*3+2]);
    }
    // Calculate pointers to current batch/camera/gaussian data
    const scalar_t *current_means_ptr    = means    + bid * N * 3 + gid * 3;
    const scalar_t *current_viewmats_ptr = viewmats + bid * C * 16 + cid * 16;
    const scalar_t *current_Ks_ptr       = Ks       + bid * C * 9 + cid * 9;

    const scalar_t *current_conics_ptr       = conics        + idx * 3;
    const scalar_t *current_v_means2d_ptr    = v_means2d     + idx * 2;
    const scalar_t *current_v_depths_ptr     = v_depths      + idx;
    const scalar_t *current_v_conics_ptr     = v_conics      + idx * 3;

    // --- VJP: Compute the inverse of the 2D covariance ---
    mat2 covar2d_inv(current_conics_ptr[0], current_conics_ptr[1],
                     current_conics_ptr[1], current_conics_ptr[2]);

    mat2 v_covar2d_inv(current_v_conics_ptr[0], current_v_conics_ptr[1] * 0.5f,
                       current_v_conics_ptr[1] * 0.5f, current_v_conics_ptr[2]);

    mat2 v_covar2d(0.f); // Initialize to zero for accumulation
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d); // Changed to scalar_t
if (idx % 100000 == 0 && DEBUG_PRINT) {        
    printf("  v_covar2d_inv (post-init): [[%f, %f], [%f, %f]]\n",
               v_covar2d_inv[0][0], v_covar2d_inv[1][0], // GLM column-major: [0][0], [1][0] is first row
               v_covar2d_inv[0][1], v_covar2d_inv[1][1]); // [0][1], [1][1] is second row
        printf("  v_covar2d (after inverse_vjp): [[%f, %f], [%f, %f]]\n",
               v_covar2d[0][0], v_covar2d[1][0],
               v_covar2d[0][1], v_covar2d[1][1]);
    }
    // --- VJP: Compensation term (if applicable) ---
    if (v_compensations != nullptr) {
        const scalar_t compensation = compensations[idx];
        const scalar_t v_compensation = v_compensations[idx];
        add_blur_vjp(eps2d, covar2d_inv, compensation, v_compensation, v_covar2d); // Changed to scalar_t
        if (idx % 100000 == 0 && DEBUG_PRINT) {        
             printf("  v_covar2d (after add_blur_vjp): [[%f, %f], [%f, %f]]\n",
                   v_covar2d[0][0], v_covar2d[1][0],
                   v_covar2d[0][1], v_covar2d[1][1]);
        }
    }

    // --- Extract Rotation (R) and Translation (t) from view matrix ---
    // Assuming viewmats is a 4x4 row-major matrix, and we extract the 3x3 rotation part
    // GLM mat3 is column-major, so this construction properly maps row-major 3x3 into GLM's column-major.
    mat3 R(current_viewmats_ptr[0], current_viewmats_ptr[4], current_viewmats_ptr[8],  // Column 0
           current_viewmats_ptr[1], current_viewmats_ptr[5], current_viewmats_ptr[9],  // Column 1
           current_viewmats_ptr[2], current_viewmats_ptr[6], current_viewmats_ptr[10]); // Column 2
    vec3 t(current_viewmats_ptr[3], current_viewmats_ptr[7], current_viewmats_ptr[11]); // Translation column

    mat3 covar;
    vec4 quat;
    vec3 scale;

    const scalar_t *current_covars_ptr = nullptr;
    const scalar_t *current_quats_ptr = nullptr;
    const scalar_t *current_scales_ptr = nullptr;

    if (covars != nullptr) {
        // If precomputed covariance is provided (symmetric 6 elements)
        current_covars_ptr = covars + bid * N * 6 + gid * 6;
        covar = mat3(
            current_covars_ptr[0], current_covars_ptr[1], current_covars_ptr[2], // Column 0 (xx, xy, xz)
            current_covars_ptr[1], current_covars_ptr[3], current_covars_ptr[4], // Column 1 (yx, yy, yz)
            current_covars_ptr[2], current_covars_ptr[4], current_covars_ptr[5]  // Column 2 (zx, zy, zz)
        );
        if (idx % 100000 == 0 && DEBUG_PRINT) {        
            printf("  Using precomputed covars. covar (world): \n");
            printf("    [%f, %f, %f]\n", covar[0][0], covar[1][0], covar[2][0]);
            printf("    [%f, %f, %f]\n", covar[0][1], covar[1][1], covar[2][1]);
            printf("    [%f, %f, %f]\n", covar[0][2], covar[1][2], covar[2][2]);
        }
    } else {
        // Compute 3D covariance from quaternion and scale
        current_quats_ptr = quats + bid * N * 4 + gid * 4;
        current_scales_ptr = scales + bid * N * 3 + gid * 3;
        quat = glm::make_vec4(current_quats_ptr);
        scale = glm::make_vec3(current_scales_ptr);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr); // Changed to scalar_t
        if (idx % 100000 == 0 && DEBUG_PRINT) {        
            printf("  Using quats and scales. quat: [%f, %f, %f, %f]\n", quat.x, quat.y, quat.z, quat.w);
            printf("  scale: [%f, %f, %f]\n", scale.x, scale.y, scale.z);
            printf("  covar (world) from quat_scale_to_covar_preci: \n");
            printf("    [%f, %f, %f]\n", covar[0][0], covar[1][0], covar[2][0]);
            printf("    [%f, %f, %f]\n", covar[0][1], covar[1][1], covar[2][1]);
            printf("    [%f, %f, %f]\n", covar[0][2], covar[1][2], covar[2][2]);
        }
    }

    // --- Transform Gaussian to Camera Space ---
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(current_means_ptr), mean_c); // Changed to scalar_t
    mat3 covar_c;
    covarW2C(R, covar, covar_c); // Changed to scalar_t

    // Extract camera intrinsics
    float fx = current_Ks_ptr[0], cx = current_Ks_ptr[2];
    float fy = current_Ks_ptr[4], cy = current_Ks_ptr[5];
    if (idx % 100000 == 0 && DEBUG_PRINT) {        
         printf("  R (viewmat rotation): \n");
        printf("    [%f, %f, %f]\n", R[0][0], R[1][0], R[2][0]);
        printf("    [%f, %f, %f]\n", R[0][1], R[1][1], R[2][1]);
        printf("    [%f, %f, %f]\n", R[0][2], R[1][2], R[2][2]);
        printf("  t (viewmat translation): [%f, %f, %f]\n", t.x, t.y, t.z);
        //printf("  mean_world_val: [%f, %f, %f]\n", mean_world_val.x, mean_world_val.y, mean_world_val.z);
        printf("  mean_c (camera space): [%f, %f, %f]\n", mean_c.x, mean_c.y, mean_c.z);
        printf("  covar_c (camera space): \n");
        printf("    [%f, %f, %f]\n", covar_c[0][0], covar_c[1][0], covar_c[2][0]);
        printf("    [%f, %f, %f]\n", covar_c[0][1], covar_c[1][1], covar_c[2][1]);
        printf("    [%f, %f, %f]\n", covar_c[0][2], covar_c[1][2], covar_c[2][2]);
    }
    mat3 v_covar_c(0.f); // Initialize to zero for accumulation
    vec3 v_mean_c(0.f);  // Initialize to zero for accumulation
    if (idx % 100000 == 0 && DEBUG_PRINT) {        
        printf("  Camera Intrinsics: fx=%f, fy=%f, cx=%f, cy=%f\n", fx, fy, cx, cy);
        printf("  v_means2d_in for proj_vjp: [%f, %f]\n", (float)current_v_means2d_ptr[0], (float)current_v_means2d_ptr[1]);
    }
    // --- VJP: Camera Projection based on model type ---
    switch (camera_model) {
        case CameraModelType::PINHOLE:
            persp_proj_vjp(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height,
                           v_covar2d, glm::make_vec2(current_v_means2d_ptr),
                           v_mean_c, v_covar_c); // Changed to scalar_t
            break;
        case CameraModelType::ORTHO:
            ortho_proj_vjp(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height,
                           v_covar2d, glm::make_vec2(current_v_means2d_ptr),
                           v_mean_c, v_covar_c); // Changed to scalar_t
            break;
        case CameraModelType::FISHEYE:
            fisheye_proj_vjp(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height,
                             v_covar2d, glm::make_vec2(current_v_means2d_ptr),
                             v_mean_c, v_covar_c); // Changed to scalar_t
            break;
    }
    if (idx % 100000 == 0 && DEBUG_PRINT) {        
        printf("  v_mean_c (after proj_vjp): [%f, %f, %f]\n", v_mean_c.x, v_mean_c.y, v_mean_c.z);
        printf("  v_covar_c (after proj_vjp): \n");
        printf("    [%f, %f, %f]\n", v_covar_c[0][0], v_covar_c[1][0], v_covar_c[2][0]);
        printf("    [%f, %f, %f]\n", v_covar_c[0][1], v_covar_c[1][1], v_covar_c[2][1]);
        printf("    [%f, %f, %f]\n", v_covar_c[0][2], v_covar_c[1][2], v_covar_c[2][2]);
    }
    // Add contribution from v_depths to the z-component of v_mean_c
    v_mean_c.z += current_v_depths_ptr[0];
    if (idx % 100000 == 0 && DEBUG_PRINT) {        
        printf("  v_mean_c (after v_depths add): [%f, %f, %f]\n", v_mean_c.x, v_mean_c.y, v_mean_c.z);
    }
    // --- VJP: Transform Gaussian from Camera to World Space ---
    vec3 v_mean_local(0.f);   // Local gradient for world mean
    mat3 v_covar_local(0.f);  // Local gradient for world covariance
    mat3 v_R_local(0.f);      // Local gradient for rotation matrix R
    vec3 v_t_local(0.f);      // Local gradient for translation vector t

    posW2C_VJP(R, t, glm::make_vec3(current_means_ptr), v_mean_c, v_R_local, v_t_local, v_mean_local); // Changed to scalar_t
    covarW2C_VJP(R, covar, v_covar_c, v_R_local, v_covar_local); // Changed to scalar_t
    if (idx % 100000 == 0 && DEBUG_PRINT) {        
        printf("  v_mean_local (after W2C_VJP): [%f, %f, %f]\n", v_mean_local.x, v_mean_local.y, v_mean_local.z);
        printf("  v_covar_local (after W2C_VJP): \n");
        printf("    [%f, %f, %f]\n", v_covar_local[0][0], v_covar_local[1][0], v_covar_local[2][0]);
        printf("    [%f, %f, %f]\n", v_covar_local[0][1], v_covar_local[1][1], v_covar_local[2][1]);
        printf("    [%f, %f, %f]\n", v_covar_local[0][2], v_covar_local[1][2], v_covar_local[2][2]);
        printf("  v_R_local (after W2C_VJP): \n");
        printf("    [%f, %f, %f]\n", v_R_local[0][0], v_R_local[1][0], v_R_local[2][0]);
        printf("    [%f, %f, %f]\n", v_R_local[0][1], v_R_local[1][1], v_R_local[2][1]);
        printf("    [%f, %f, %f]\n", v_R_local[0][2], v_R_local[1][2], v_R_local[2][2]);
        printf("  v_t_local (after W2C_VJP): [%f, %f, %f]\n", v_t_local.x, v_t_local.y, v_t_local.z);
    }
    // Get warp context for dynamic reductions
    unsigned int warp_thread_id = threadIdx.x % 64;
    unsigned long long warp_active_mask = __activemask();
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    // --- DENSE GRADIENT ACCUMULATION (Gaussian-specific parameters) ---
    // This path uses warp-level reductions and atomic adds to global memory.
    if (!sparse_grad) {
        #if USE_ROCM
        // Manual emulation of labeled_partition + reduce for Gaussian-related gradients
        // This calculates the sum within the warp for a given GID.
        if (v_means != nullptr) {
            manual_dynamic_reduce_sum_vec3(
                v_mean_local,
                gid, // Use GID as the label for reduction
                warp_thread_id,
                warp_active_mask
            );
        if (idx % 100000 == 0 && DEBUG_PRINT) {        
            printf("  v_mean_local (after manual dynamic reduce sum): [%f, %f, %f]\n", v_mean_local.x, v_mean_local.y, v_mean_local.z);
        }
            // Elect a leader for atomic write to global memory.
            unsigned long long my_gid_mask = 0;
            for (int i = 0; i < 64; ++i) {
                long long lane_gid_temp = __shfl_sync(warp_active_mask, gid, i);
                if ((warp_active_mask & (1ULL << i)) && (lane_gid_temp == gid)) {
                    my_gid_mask |= (1ULL << i);
                }
            }
            int my_warp_leader_lane_id = get_leader_lane_id(my_gid_mask);

            if (warp_thread_id == my_warp_leader_lane_id) {
                scalar_t* target_v_means_ptr = v_means + bid * N * 3 + gid * 3;
                // --- FIX: Cast float to scalar_t for unsafeAtomicAdd ---
                unsafeAtomicAdd(&(target_v_means_ptr[0]), static_cast<scalar_t>(v_mean_local.x));
                unsafeAtomicAdd(&(target_v_means_ptr[1]), static_cast<scalar_t>(v_mean_local.y));
                unsafeAtomicAdd(&(target_v_means_ptr[2]), static_cast<scalar_t>(v_mean_local.z));
            }
        }
        if (v_covars != nullptr) {
            manual_dynamic_reduce_sum_mat3(
                v_covar_local,
                gid, // Use GID as the label for reduction
                warp_thread_id,
                warp_active_mask
            );

            unsigned long long my_gid_mask = 0;
            for (int i = 0; i < 64; ++i) {
                long long lane_gid_temp = __shfl_sync(warp_active_mask, gid, i);
                if ((warp_active_mask & (1ULL << i)) && (lane_gid_temp == gid)) {
                    my_gid_mask |= (1ULL << i);
                }
            }
            int my_warp_leader_lane_id = get_leader_lane_id(my_gid_mask);

            if (warp_thread_id == my_warp_leader_lane_id) {
                scalar_t* target_v_covars_ptr = v_covars + bid * N * 6 + gid * 6;
                // Accumulate unique elements of the symmetric covariance gradient
                // --- FIX: Cast float to scalar_t for unsafeAtomicAdd ---
                unsafeAtomicAdd(target_v_covars_ptr,     static_cast<scalar_t>(v_covar_local[0][0]));
                unsafeAtomicAdd(target_v_covars_ptr + 1, static_cast<scalar_t>(v_covar_local[0][1] + v_covar_local[1][0])); // xy and yx
                unsafeAtomicAdd(target_v_covars_ptr + 2, static_cast<scalar_t>(v_covar_local[0][2] + v_covar_local[2][0])); // xz and zx
                unsafeAtomicAdd(target_v_covars_ptr + 3, static_cast<scalar_t>(v_covar_local[1][1]));
                unsafeAtomicAdd(target_v_covars_ptr + 4, static_cast<scalar_t>(v_covar_local[1][2] + v_covar_local[2][1])); // yz and zy
                unsafeAtomicAdd(target_v_covars_ptr + 5, static_cast<scalar_t>(v_covar_local[2][2]));
            }
        } else { // Handle v_quats and v_scales if covars are not provided
            vec4 v_quat_local(0.f);   // Local gradient for quaternion
            vec3 v_scale_local(0.f);  // Local gradient for scale

            mat3 rotmat = quat_to_rotmat(quat);
            quat_scale_to_covar_vjp(
                quat, scale, rotmat, v_covar_local, v_quat_local, v_scale_local
            );

            manual_dynamic_reduce_sum_vec4(v_quat_local, gid, warp_thread_id, warp_active_mask);
            manual_dynamic_reduce_sum_vec3(v_scale_local, gid, warp_thread_id, warp_active_mask);

            unsigned long long my_gid_mask = 0;
            for (int i = 0; i < 64; ++i) {
                long long lane_gid_temp = __shfl_sync(warp_active_mask, gid, i);
                if ((warp_active_mask & (1ULL << i)) && (lane_gid_temp == gid)) {
                    my_gid_mask |= (1ULL << i);
                }
            }
            int my_warp_leader_lane_id = get_leader_lane_id(my_gid_mask);

            if (warp_thread_id == my_warp_leader_lane_id) {
                scalar_t* target_v_quats_ptr = v_quats + bid * N * 4 + gid * 4;
                scalar_t* target_v_scales_ptr = v_scales + bid * N * 3 + gid * 3;
                // --- FIX: Cast float to scalar_t for unsafeAtomicAdd ---
                unsafeAtomicAdd(target_v_quats_ptr,     static_cast<scalar_t>(v_quat_local.x));
                unsafeAtomicAdd(target_v_quats_ptr + 1, static_cast<scalar_t>(v_quat_local.y));
                unsafeAtomicAdd(target_v_quats_ptr + 2, static_cast<scalar_t>(v_quat_local.z));
                unsafeAtomicAdd(target_v_quats_ptr + 3, static_cast<scalar_t>(v_quat_local.w));
                unsafeAtomicAdd(target_v_scales_ptr,    static_cast<scalar_t>(v_scale_local.x));
                unsafeAtomicAdd(target_v_scales_ptr + 1, static_cast<scalar_t>(v_scale_local.y));
                unsafeAtomicAdd(target_v_scales_ptr + 2, static_cast<scalar_t>(v_scale_local.z));
            }
        }
        #else // USE_ROCM is 0, use Cooperative Groups Labeled Partition
        // Preferred path for modern CUDA: using cg::labeled_partition
        auto warp_group_g = cg::labeled_partition(warp, gid);
        // No explicit is_valid() check needed here as `labeled_partition` guarantees valid sub-groups for active threads.

        if (v_means != nullptr) {
            warpSum(v_mean_local, warp_group_g);
            if (idx % 100000 == 0 && DEBUG_PRINT) {        
                printf("  v_mean_local (after c gWarpSum): [%f, %f, %f]\n", v_mean_local.x, v_mean_local.y, v_mean_local.z);
            }
            if (warp_group_g.thread_rank() == 0) { // Only leader of the group writes
                scalar_t* target_v_means_ptr = v_means + bid * N * 3 + gid * 3;
                // --- FIX: Cast float to scalar_t for unsafeAtomicAdd ---
                unsafeAtomicAdd(&(target_v_means_ptr[0]), static_cast<scalar_t>(v_mean_local.x));
                unsafeAtomicAdd(&(target_v_means_ptr[1]), static_cast<scalar_t>(v_mean_local.y));
                unsafeAtomicAdd(&(target_v_means_ptr[2]), static_cast<scalar_t>(v_mean_local.z));
            }
        }
        if (v_covars != nullptr) {
            warpSum(v_covar_local, warp_group_g);
            if (warp_group_g.thread_rank() == 0) { // Only leader of the group writes
                scalar_t* target_v_covars_ptr = v_covars + bid * N * 6 + gid * 6;
                // Accumulate unique elements of the symmetric covariance gradient
                // --- FIX: Cast float to scalar_t for unsafeAtomicAdd ---
                unsafeAtomicAdd(target_v_covars_ptr,     static_cast<scalar_t>(v_covar_local[0][0]));
                unsafeAtomicAdd(target_v_covars_ptr + 1, static_cast<scalar_t>(v_covar_local[0][1] + v_covar_local[1][0]));
                unsafeAtomicAdd(target_v_covars_ptr + 2, static_cast<scalar_t>(v_covar_local[0][2] + v_covar_local[2][0]));
                unsafeAtomicAdd(target_v_covars_ptr + 3, static_cast<scalar_t>(v_covar_local[1][1]));
                unsafeAtomicAdd(target_v_covars_ptr + 4, static_cast<scalar_t>(v_covar_local[1][2] + v_covar_local[2][1]));
                unsafeAtomicAdd(target_v_covars_ptr + 5, static_cast<scalar_t>(v_covar_local[2][2]));
            }
        } else {
            // Directly output gradients w.r.t. the quaternion and scale
            mat3 rotmat = quat_to_rotmat(quat);
            vec4 v_quat_local(0.f);
            vec3 v_scale_local(0.f);
            quat_scale_to_covar_vjp(
                quat, scale, rotmat, v_covar_local, v_quat_local, v_scale_local
            );

            warpSum(v_quat_local, warp_group_g);
            warpSum(v_scale_local, warp_group_g);
            if (warp_group_g.thread_rank() == 0) { // Only leader of the group writes
                scalar_t* target_v_quats_ptr = v_quats + bid * N * 4 + gid * 4;
                scalar_t* target_v_scales_ptr = v_scales + bid * N * 3 + gid * 3;
                // --- FIX: Cast float to scalar_t for unsafeAtomicAdd ---
                unsafeAtomicAdd(target_v_quats_ptr,     static_cast<scalar_t>(v_quat_local.x));
                unsafeAtomicAdd(target_v_quats_ptr + 1, static_cast<scalar_t>(v_quat_local.y));
                unsafeAtomicAdd(target_v_quats_ptr + 2, static_cast<scalar_t>(v_quat_local.z));
                unsafeAtomicAdd(target_v_quats_ptr + 3, static_cast<scalar_t>(v_quat_local.w));
                unsafeAtomicAdd(target_v_scales_ptr,    static_cast<scalar_t>(v_scale_local.x));
                unsafeAtomicAdd(target_v_scales_ptr + 1, static_cast<scalar_t>(v_scale_local.y));
                unsafeAtomicAdd(target_v_scales_ptr + 2, static_cast<scalar_t>(v_scale_local.z));
            }
        }
        #endif
    } else {
        // --- SPARSE GRADIENT OUTPUT (per-thread) ---
        // This path is for when each thread outputs its own gradient, no aggregation needed.
        if (v_means != nullptr) {
            scalar_t* target_v_means_ptr = v_means + idx * 3;
            target_v_means_ptr[0] = v_mean_local.x;
            target_v_means_ptr[1] = v_mean_local.y;
            target_v_means_ptr[2] = v_mean_local.z;
        }
        if (v_covars != nullptr) {
            scalar_t* target_v_covars_ptr = v_covars + idx * 6;
            target_v_covars_ptr[0] = v_covar_local[0][0];
            target_v_covars_ptr[1] = v_covar_local[0][1] + v_covar_local[1][0];
            target_v_covars_ptr[2] = v_covar_local[0][2] + v_covar_local[2][0];
            target_v_covars_ptr[3] = v_covar_local[1][1];
            target_v_covars_ptr[4] = v_covar_local[1][2] + v_covar_local[2][1];
            target_v_covars_ptr[5] = v_covar_local[2][2];
        } else {
            mat3 rotmat = quat_to_rotmat(quat);
            vec4 v_quat_local(0.f);
            vec3 v_scale_local(0.f);
            quat_scale_to_covar_vjp(
                quat, scale, rotmat, v_covar_local, v_quat_local, v_scale_local
            );
            scalar_t* target_v_quats_ptr = v_quats + idx * 4;
            scalar_t* target_v_scales_ptr = v_scales + idx * 3;
            target_v_quats_ptr[0] = v_quat_local.x;
            target_v_quats_ptr[1] = v_quat_local.y;
            target_v_quats_ptr[2] = v_quat_local.z;
            target_v_quats_ptr[3] = v_quat_local.w;
            target_v_scales_ptr[0] = v_scale_local.x;
            target_v_scales_ptr[1] = v_scale_local.y;
            target_v_scales_ptr[2] = v_scale_local.z;
        }
    }

    // --- GRADIENT ACCUMULATION for v_viewmats (Camera-specific) ---
    // v_viewmats is always dense, so atomic adds are needed regardless of sparse_grad.
    if (v_viewmats != nullptr) {
        #if USE_ROCM
        // Manual emulation for Camera-related gradients
        manual_dynamic_reduce_sum_mat3(v_R_local, cid, warp_thread_id, warp_active_mask);
        manual_dynamic_reduce_sum_vec3(v_t_local, cid, warp_thread_id, warp_active_mask);

        unsigned long long my_cid_mask = 0;
        for (int i = 0; i < 64; ++i) {
            long long lane_cid_temp = __shfl_sync(warp_active_mask, cid, i);
            if ((warp_active_mask & (1ULL << i)) && (lane_cid_temp == cid)) {
                my_cid_mask |= (1ULL << i);
            }
        }
        int my_warp_leader_lane_id_cid = get_leader_lane_id(my_cid_mask);

        if (warp_thread_id == my_warp_leader_lane_id_cid) {
            scalar_t* target_v_viewmats_ptr = v_viewmats + bid * C * 16 + cid * 16;
            for (uint32_t i = 0; i < 3; i++) { // rows (0, 1, 2)
                for (uint32_t j = 0; j < 3; j++) { // cols (0, 1, 2) - rotation part
                    // v_R_local is GLM (column-major). target_v_viewmats_ptr is row-major.
                    // Access [col][row] for v_R_local to transpose to row-major output.
                    // --- FIX: Cast float to scalar_t for unsafeAtomicAdd ---
                    unsafeAtomicAdd(target_v_viewmats_ptr + i * 4 + j, static_cast<scalar_t>(v_R_local[j][i]));
                }
                // Add translation components to the 4th column (index 3)
                // --- FIX: Cast float to scalar_t for unsafeAtomicAdd ---
                if (i == 0) unsafeAtomicAdd(target_v_viewmats_ptr + i * 4 + 3, static_cast<scalar_t>(v_t_local.x));
                if (i == 1) unsafeAtomicAdd(target_v_viewmats_ptr + i * 4 + 3, static_cast<scalar_t>(v_t_local.y));
                if (i == 2) unsafeAtomicAdd(target_v_viewmats_ptr + i * 4 + 3, static_cast<scalar_t>(v_t_local.z));
            }
        }
        #else // USE_ROCM is 0, use Cooperative Groups Labeled Partition
        auto warp_group_c = cg::labeled_partition(warp, cid);

        warpSum(v_R_local, warp_group_c);
        warpSum(v_t_local, warp_group_c);
        if (warp_group_c.thread_rank() == 0) { // Only leader of the group writes
            scalar_t* target_v_viewmats_ptr = v_viewmats + bid * C * 16 + cid * 16;
            for (uint32_t i = 0; i < 3; i++) { // rows (0, 1, 2)
                for (uint32_t j = 0; j < 3; j++) { // cols (0, 1, 2) - rotation part
                    // --- FIX: Cast float to scalar_t for unsafeAtomicAdd ---
                    unsafeAtomicAdd(target_v_viewmats_ptr + i * 4 + j, static_cast<scalar_t>(v_R_local[j][i]));
                }
                // Add translation components to the 4th column (index 3)
                // --- FIX: Cast float to scalar_t for unsafeAtomicAdd ---
                if (i == 0) unsafeAtomicAdd(target_v_viewmats_ptr + i * 4 + 3, static_cast<scalar_t>(v_t_local.x));
                if (i == 1) unsafeAtomicAdd(target_v_viewmats_ptr + i * 4 + 3, static_cast<scalar_t>(v_t_local.y));
                if (i == 2) unsafeAtomicAdd(target_v_viewmats_ptr + i * 4 + 3, static_cast<scalar_t>(v_t_local.z));
            }
        }
        #endif
    }
}

void launch_projection_ewa_3dgs_packed_bwd_kernel(
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6]
    const at::optional<at::Tensor> quats,  // [..., N, 4]
    const at::optional<at::Tensor> scales, // [..., N, 3]
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor batch_ids,                   // [nnz]
    const at::Tensor camera_ids,                  // [nnz]
    const at::Tensor gaussian_ids,                // [nnz]
    const at::Tensor conics,                      // [nnz, 3]
    const at::optional<at::Tensor> compensations, // [nnz] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [nnz, 2]
    const at::Tensor v_depths,                      // [nnz]
    const at::Tensor v_conics,                      // [nnz, 3]
    const at::optional<at::Tensor> v_compensations, // [nnz] optional
    const bool sparse_grad,
    // grad inputs
    at::Tensor v_means,                 // [..., N, 3] or [nnz, 3]
    at::optional<at::Tensor> v_covars,  // [..., N, 6] or [nnz, 6] Optional
    at::optional<at::Tensor> v_quats,   // [..., N, 4] or [nnz, 4] Optional
    at::optional<at::Tensor> v_scales,  // [..., N, 3] or [nnz, 3] Optional
    at::optional<at::Tensor> v_viewmats // [..., C, 4, 4] Optional
) {
    uint32_t N = means.size(-2);          // number of gaussians
    uint32_t C = viewmats.size(-3);       // number of cameras
    uint32_t B = means.numel() / (N * 3); // number of batches
    uint32_t nnz = batch_ids.size(0);

    dim3 threads(256);
    dim3 grid((nnz + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (nnz == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_packed_bwd_kernel",
        [&]() {
            projection_ewa_3dgs_packed_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   GET_CURRENT_STREAM()>>>(
                    B, 
                    C,
                    N,
                    nnz,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    covars.has_value() ? nullptr
                                       : quats.value().data_ptr<scalar_t>(),
                    covars.has_value() ? nullptr
                                       : scales.value().data_ptr<scalar_t>(),
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    camera_model,
                    batch_ids.data_ptr<int64_t>(),
                    camera_ids.data_ptr<int64_t>(),
                    gaussian_ids.data_ptr<int64_t>(),
                    conics.data_ptr<scalar_t>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    v_means2d.data_ptr<scalar_t>(),
                    v_depths.data_ptr<scalar_t>(),
                    v_conics.data_ptr<scalar_t>(),
                    v_compensations.has_value()
                        ? v_compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    sparse_grad,
                    v_means.data_ptr<scalar_t>(),
                    covars.has_value() ? v_covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    covars.has_value() ? nullptr
                                       : v_quats.value().data_ptr<scalar_t>(),
                    covars.has_value() ? nullptr
                                       : v_scales.value().data_ptr<scalar_t>(),
                    v_viewmats.has_value()
                        ? v_viewmats.value().data_ptr<scalar_t>()
                        : nullptr
                );
        }
    );
}

} // namespace gsplat
