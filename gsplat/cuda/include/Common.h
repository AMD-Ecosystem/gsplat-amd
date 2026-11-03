#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>

#ifndef USE_ROCM
#include <c10/cuda/CUDAStream.h> // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDACachingAllocator.h>
#include <cooperative_groups.h>
#include <c10/hip/CUDAGuard.h>
#else
#include <c10/hip/HIPStream.h> // at::hip::getCurrentHIPStream
#include <c10/hip/HIPCachingAllocator.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>
#include <c10/core/DeviceGuard.h>
#include <ATen/DeviceGuard.h>
#endif

namespace gsplat {

//
// Some Macros.
//
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

#ifndef USE_ROCM
#define GET_CURRENT_STREAM() at::cuda::getCurrentCUDAStream()
#define DEVICE_GUARD(_ten)                                                     \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));
// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
// handle the temporary storage and 'twice' calls for cub API
#else
#define cub hipcub
#define GET_CURRENT_STREAM() at::hip::getCurrentHIPStream()
#define DEVICE_GUARD(_ten)
    //const c10::OptionalDeviceGuard device_guard(at::device_of(_ten));                                                   \
    const at::hip::OptionalHIPGuard device_guard(device_of(_ten));
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaSuccess hipSuccess
#endif

//
// Convenience typedefs for CUDA types
//
using vec2 = glm::vec<2, float>;
using vec3 = glm::vec<3, float>;
using vec4 = glm::vec<4, float>;
using mat2 = glm::mat<2, 2, float>;
using mat3 = glm::mat<3, 3, float>;
using mat4 = glm::mat<4, 4, float>;
using mat3x2 = glm::mat<3, 2, float>;

//
// Legacy Camera Types
//
enum CameraModelType {
    PINHOLE = 0,
    ORTHO = 1,
    FISHEYE = 2,
    FTHETA = 3,
};

#define N_THREADS_PACKED 256
#define ALPHA_THRESHOLD (1.f / 255.f)

} // namespace gsplat