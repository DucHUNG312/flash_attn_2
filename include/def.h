#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define FA_DEVICE __forceinline__ __device__
#define FA_HOST_DEVICE __forceinline__ __host__ __device__
#define FA_DEVICE_CONSTEXPR __forceinline__ __device__ constexpr
#define FA_HOST_DEVICE_CONSTEXPR __forceinline__ __device__ constexpr

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define FA_DEVICE_SUPPORTED
#endif