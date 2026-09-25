#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>
#include <torch/torch.h>

#ifndef USE_ROCM
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <c10/hip/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>

#else
#include <hip/hip_cooperative_groups.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/hip/Atomic.cuh>
#include <hipcub/hipcub.hpp>
#include <hipcub/block/block_reduce.hpp>
#include <rocprim/warp/warp_reduce.hpp>
#endif

namespace gsplat {

#ifndef USE_ROCM
// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...)                                                 \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)
#else
#define cub hipcub

// PyTorch ROCm allocator selection
// PyTorch 2.11 has a breaking API change in which c10::hip::HIPCachingAllocator 
// was removed and unified into c10::cuda::CUDACachingAllocator.
#if (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 11)
#define GSPLAT_ROCM_ALLOCATOR ::c10::cuda::CUDACachingAllocator
#else
#define GSPLAT_ROCM_ALLOCATOR ::c10::hip::HIPCachingAllocator
#endif

#define CUB_WRAPPER(func, ...)                                                 \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        auto res = func(nullptr, temp_storage_bytes, __VA_ARGS__);             \
	auto& caching_allocator = *GSPLAT_ROCM_ALLOCATOR::get();                                               \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        res = func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);       \
	assert(res == hipSuccess);                                             \
    } while (false)
#endif
} // namespace gsplat
