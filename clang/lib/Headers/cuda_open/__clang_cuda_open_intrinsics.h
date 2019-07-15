//===--------- cuda_open/__clang_cuda_open_intrinsics.h  ------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef __CLANG_CUDA_OPEN_INTRINSICS_H__
#define __CLANG_CUDA_OPEN_INTRINSICS_H__
#ifndef __CUDA__
#error "This file is for CUDA compilation only."
#endif
#ifndef __USE_OPEN_HEADERS__
#error "This file requires -D__USE_OPEN_HEADERS__"
#endif

#ifdef __cplusplus
__device__ int __shfl(int input, int lane, int width = warpSize);
__device__ int __shfl_up(int input, unsigned int lane_delta,
                         int width = warpSize);
__device__ int __shfl_down(int input, unsigned int lane_delta,
                           int width = warpSize);
__device__ int __shfl_xor(int input, int lane_mask, int width = warpSize);
__device__ float __shfl(float input, int lane, int width = warpSize);
__device__ float __shfl_up(float input, unsigned int lane_delta,
                           int width = warpSize);
__device__ float __shfl_down(float input, unsigned int lane_delta,
                             int width = warpSize);
__device__ float __shfl_xor(float input, int lane_mask, int width = warpSize);
#else
__device__ int __shfl(int input, int lane, int width);
__device__ int __shfl_up(int input, unsigned int lane_delta, int width);
__device__ int __shfl_down(int input, unsigned int lane_delta, int width);
__device__ int __shfl_xor(int input, int lane_mask, int width);
__device__ float __shfl(float input, int lane, int width);
__device__ float __shfl_up(float input, unsigned int lane_delta, int width);
__device__ float __shfl_down(float input, unsigned int lane_delta, int width);
__device__ float __shfl_xor(float input, int lane_mask, int width);
#endif //__cplusplus

#endif
