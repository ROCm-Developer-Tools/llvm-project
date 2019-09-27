//===---- cuda_shim.h - Minimal CUDA API to compile as -x HIP ---- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a sufficient implementation of the open cuda API to
// compile the amdgcn deviceRTL as hip, without using hip_runtime.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_SHIM_H
#define __CUDA_SHIM_H

#ifndef __AMDGCN__
#error "amdgcn cuda_shim.h expects to be compiled under __AMDGCN__"
#endif

#ifndef __device__
#define __device__ __attribute__((device))
#endif

#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif

#ifndef __noinline__
#define __noinline__ __attribute__((noinline))
#endif

#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline))
#endif

#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif

// device libc declarations
extern "C" {
__device__ void __assert_fail(const char *, const char *, unsigned int,
                              const char *);
__device__ int printf(const char *, ...);
}

#ifndef __OVERL__
#define __OVERL__ __attribute__((device, always_inline, overloadable)) const
#endif

#ifndef __NOOVL__
#define __NOOVL__ extern "C" __attribute__((device, always_inline)) const
#endif

// shuffle
static constexpr int warpSize = 64;
__OVERL__ int __shfl_down(int a, unsigned int b, int c);
__OVERL__
int __shfl(int var, int src_lane, int width = warpSize);
__OVERL__
unsigned int __shfl(unsigned int var, int src_lane, int width = warpSize);

// Declare here, defined in IR as shims around nvvm.member.gl et al
// Using asm() to rename the symbol works, but marks the symbols as convergent
// which changes the CFG
extern "C" {
__device__ void __nvvm_membar_gl(void);
__device__ void __nvvm_membar_cta(void);
__device__ void __nvvm_membar_sys(void);
__device__ uint32_t __nvvm_read_ptx_sreg_tid_x(void);
__device__ uint32_t __nvvm_read_ptx_sreg_tid_y(void);
__device__ uint32_t __nvvm_read_ptx_sreg_tid_z(void);
__device__ uint32_t __nvvm_read_ptx_sreg_ctaid_x(void);
__device__ uint32_t __nvvm_read_ptx_sreg_ctaid_y(void);
__device__ uint32_t __nvvm_read_ptx_sreg_ctaid_z(void);
__device__ uint32_t __nvvm_read_ptx_sreg_ntid_x(void);
__device__ uint32_t __nvvm_read_ptx_sreg_ntid_y(void);
__device__ uint32_t __nvvm_read_ptx_sreg_ntid_z(void);
__device__ uint32_t __nvvm_read_ptx_sreg_nctaid_x(void);
__device__ uint32_t __nvvm_read_ptx_sreg_nctaid_y(void);
__device__ uint32_t __nvvm_read_ptx_sreg_nctaid_z(void);
}

__device__ __forceinline__ static void __threadfence(void) {
  __nvvm_membar_gl();
}
__device__ __forceinline__ static void __threadfence_block(void) {
  __nvvm_membar_cta();
}
__device__ __forceinline__ static void __threadfence_system(void) {
  __nvvm_membar_sys();
}
__device__ __forceinline__ static long long __clock64(void) {
  return __builtin_amdgcn_s_memrealtime(); // TODO: Support older architectures?
}

extern "C" {
__device__ int __nv_popcll(long long);
__device__ int __nv_ffsll(long long);
__device__ int __nv_min(int, int);
}

__device__ __forceinline__ static int __popcll(long long __a) {
  return __nv_popcll(__a);
}
__device__ __forceinline__ static int __ffsll(long long __a) {
  return __nv_ffsll(__a);
}
__device__ __forceinline__ static int min(int __a, int __b) {
  return __nv_min(__a, __b);
}

// atomics shim
#ifdef __cplusplus
extern "C" {
#endif

__OVERL__ unsigned atomicAdd(unsigned *address, unsigned val);
__OVERL__ int atomicAdd(int *address, int val);
__OVERL__ unsigned long long atomicAdd(unsigned long long *address,
                                       unsigned long long val);

__OVERL__ unsigned atomicInc(unsigned *address);
__OVERL__ unsigned atomicInc(unsigned *address, unsigned max);
__OVERL__ int atomicInc(int *address);

__OVERL__ int atomicMax(int *address, int val);
__OVERL__ unsigned atomicMax(unsigned *address, unsigned val);
__OVERL__ unsigned long long atomicMax(unsigned long long *address,
                                       unsigned long long val);

__OVERL__ int atomicExch(int *address, int val);
__OVERL__ unsigned atomicExch(unsigned *address, unsigned val);
__OVERL__ unsigned long long atomicExch(unsigned long long *address,
                                        unsigned long long val);

__OVERL__ unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val);
__OVERL__ int atomicCAS(int *address, int compare, int val);
__OVERL__ unsigned long long atomicCAS(unsigned long long *address,
                                       unsigned long long compare,
                                       unsigned long long val);

#ifdef __cplusplus
}
#endif

#define __CUDA_DEVICE_BUILTIN(FIELD, INTRINSIC)                                \
  __declspec(property(get = __fetch_builtin_##FIELD)) unsigned int FIELD;      \
  static inline __attribute__((always_inline))                                 \
      __attribute__((device)) unsigned int __fetch_builtin_##FIELD(void) {     \
    return INTRINSIC;                                                          \
  }

#if __cplusplus >= 201103L
#define __DELETE = delete
#else
#define __DELETE
#endif

#define __CUDA_DISALLOW_BUILTINVAR_ACCESS(TypeName)                            \
  __attribute__((device)) TypeName() __DELETE;                                 \
  __attribute__((device)) TypeName(const TypeName &) __DELETE;                 \
  __attribute__((device)) void operator=(const TypeName &) const __DELETE;     \
  __attribute__((device)) TypeName *operator&() const __DELETE

struct __cuda_builtin_threadIdx_t {
  __CUDA_DEVICE_BUILTIN(x, __nvvm_read_ptx_sreg_tid_x());
  __CUDA_DEVICE_BUILTIN(y, __nvvm_read_ptx_sreg_tid_y());
  __CUDA_DEVICE_BUILTIN(z, __nvvm_read_ptx_sreg_tid_z());

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_threadIdx_t);
};

struct __cuda_builtin_blockIdx_t {
  __CUDA_DEVICE_BUILTIN(x, __nvvm_read_ptx_sreg_ctaid_x());
  __CUDA_DEVICE_BUILTIN(y, __nvvm_read_ptx_sreg_ctaid_y());
  __CUDA_DEVICE_BUILTIN(z, __nvvm_read_ptx_sreg_ctaid_z());

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockIdx_t);
};

struct __cuda_builtin_blockDim_t {
  __CUDA_DEVICE_BUILTIN(x, __nvvm_read_ptx_sreg_ntid_x());
  __CUDA_DEVICE_BUILTIN(y, __nvvm_read_ptx_sreg_ntid_y());
  __CUDA_DEVICE_BUILTIN(z, __nvvm_read_ptx_sreg_ntid_z());

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockDim_t);
};

struct __cuda_builtin_gridDim_t {
  __CUDA_DEVICE_BUILTIN(x, __nvvm_read_ptx_sreg_nctaid_x());
  __CUDA_DEVICE_BUILTIN(y, __nvvm_read_ptx_sreg_nctaid_y());
  __CUDA_DEVICE_BUILTIN(z, __nvvm_read_ptx_sreg_nctaid_z());

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_gridDim_t);
};

#define __CUDA_BUILTIN_VAR                                                     \
  extern const __attribute__((device)) __attribute__((weak))
__CUDA_BUILTIN_VAR __cuda_builtin_threadIdx_t threadIdx;
__CUDA_BUILTIN_VAR __cuda_builtin_blockIdx_t blockIdx;
__CUDA_BUILTIN_VAR __cuda_builtin_blockDim_t blockDim;
__CUDA_BUILTIN_VAR __cuda_builtin_gridDim_t gridDim;

#endif
