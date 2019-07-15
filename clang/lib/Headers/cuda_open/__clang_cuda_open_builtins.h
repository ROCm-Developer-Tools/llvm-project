//===--------- cuda_open/__clang_cuda_open_builtins.h ---------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef __CLANG_CUDA_OPEN_BUILTINS_HPP__
#define __CLANG_CUDA_OPEN_BUILTINS_HPP__
#ifndef __CUDA__
#error "This file is for CUDA compilation only."
#endif
#define __DEVICE__ static __device__ __inline__ __attribute__((always_inline))
#define __HOST_DEVICE__                                                        \
  static __host__ __device__ __inline__ __attribute__((always_inline))

__HOST_DEVICE__ float4 make_float4(float x, float y, float z, float w) {
  float4 result;
  result.x = x;
  result.y = y;
  result.z = z;
  result.w = w;
  return result;
};

__HOST_DEVICE__ float2 make_float2(float x, float y) {
  float2 result;
  result.x = x;
  result.y = y;
  return result;
};

__HOST_DEVICE__ double2 make_double2(double x, double y) {
  double2 result;
  result.x = x;
  result.y = y;
  return result;
}

__DEVICE__ int __amdgcn_mbcnt_lo(int mask,
                                 int src) __asm("llvm.amdgcn.mbcnt.lo");
__DEVICE__ int __amdgcn_mbcnt_hi(int mask,
                                 int src) __asm("llvm.amdgcn.mbcnt.hi");

__DEVICE__ int __lane_id(void) [[hc]] {
  int lo = __amdgcn_mbcnt_lo(-1, 0);
  return __amdgcn_mbcnt_hi(-1, lo);
}

extern "C" {
__device__ inline int __amdgcn_ds_bpermute(int, int);
}

union __u {
  int i;
  unsigned int u;
  float f;
};

__DEVICE__ int __shfl_down(int var, const unsigned int delta,
                           const int width = 32) {
  int self = __lane_id();
  int index = self + delta;
  index = ((self & (width - 1)) + delta) >= width ? self : index;
  return __amdgcn_ds_bpermute(index << 2, var);
}

//__DEVICE__ float __shfl_down(float var, unsigned int delta, const int
// width=32) {
//  __u tmp; tmp.f = var;
//  tmp.i = __shfl_down(tmp.i, delta, width);
//  return tmp.f;
//}

__DEVICE__ int __shfl_xor(int var, int laneMask, int width = 32) {
  int self = __lane_id();
  int index = self ^ laneMask;
  index = index >= ((self + width) & ~(width - 1)) ? self : index;
  return __amdgcn_ds_bpermute(index << 2, var);
}
/*
__DEVICE__ float __shfl_xor(float var, int laneMask, int width32) {
    __u tmp; tmp.f = var;
    tmp.i = __shfl_xor(tmp.i, laneMask, width);
    return tmp.f;
}
*/

#undef __DEVICE__
#undef __HOST_DEVICE__
#ifndef __USE_OPEN_HEADERS__
#error "This file requires -D__USE_OPEN_HEADERS__"
#endif

#endif
