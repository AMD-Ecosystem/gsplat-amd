#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>

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

// gfx1151 (RDNA 3.5 / Strix Halo) is wave32, while this fork was written for
// CDNA wave64. WARP_SIZE is the one compile-time knob that makes the reduction /
// tiling code correct on BOTH: 64 on CDNA, 32 on RDNA(gfx11/12), 32 on NVIDIA.
//
// We key off the __gfx9__ architecture predicate, NOT __AMDGCN_WAVEFRONT_SIZE:
// only the gfx9 family (CDNA / GCN) is wave64; every gfx10/11/12 (RDNA) target is
// wave32. AMD's rocPRIM maintainers recommend this arch predicate as the durable
// form (ROCm/ROCm#4121) because __AMDGCN_WAVEFRONT_SIZE is deprecated as of ROCm
// 7.0.2 and slated for removal, and warpSize is not a host-side constant (this
// constant is also read in the host-side launch_* sizing code). The predicate is
// a compile-time macro evaluated per --offload-arch, so the SAME source compiles
// to a correct wave64 object on CDNA and a correct wave32 object on gfx1151.
#ifdef USE_ROCM
  #if defined(__gfx900__) || defined(__gfx906__) || defined(__gfx908__) ||      \
      defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) ||      \
      defined(__gfx942__) || defined(__gfx950__)
    constexpr int WARP_SIZE = 64;   // CDNA / GCN (gfx9) is wave64
  #else
    constexpr int WARP_SIZE = 32;   // RDNA (gfx10/11/12), incl. gfx1151, is wave32
  #endif
#else
    constexpr int WARP_SIZE = 32;   // NVIDIA
#endif

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
#define CUB_WRAPPER(func, ...)                                                 \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        auto res = func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::hip::HIPCachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        res = func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);  \
	assert(res == hipSuccess);                                            \
    } while (false)
#endif
} // namespace gsplat
