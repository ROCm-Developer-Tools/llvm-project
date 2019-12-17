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

// device libc
extern "C" {
__device__ __attribute__((noreturn)) void
__assertfail(const char *, const char *, unsigned, const char *, size_t);
__device__ static inline void __assert_fail(const char *__message,
                                            const char *__file,
                                            unsigned int __line,
                                            const char *__function) {
  __assertfail(__message, __file, __line, __function, sizeof(char));
}
__device__ int printf(const char *, ...);
}

#ifndef __OVERL__
#define __OVERL__ __attribute__((device, always_inline, overloadable)) const
#endif

// shuffle
static constexpr int warpSize = 64;
__OVERL__ int __shfl_down(int a, unsigned int b, int c);
__OVERL__
int __shfl(int var, int src_lane, int width = warpSize);

__device__ static long long __clock64(void) {
#if __AMDGCN__ > 800
  return __builtin_amdgcn_s_memrealtime();
#else
  __device__ long long __llvm_amdgcn_s_memrealtime(void);
  return __llvm_amdgcn_s_memrealtime();
#endif
}

__device__ static int32_t min(int32_t __a, int32_t __b) {
  return __a < __b ? __a : __b;
}
__device__ static uint32_t min(uint32_t __a, uint32_t __b) {
  return __a < __b ? __a : __b;
}
__device__ static uint32_t min(uint32_t __a, int32_t __b) {
  // resolve ambiguous call
  return min(__a, (uint32_t)__b);
}


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
  __CUDA_DEVICE_BUILTIN(x, __builtin_amdgcn_workitem_id_x());

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_threadIdx_t);
};

struct __cuda_builtin_blockIdx_t {
  __CUDA_DEVICE_BUILTIN(x, __builtin_amdgcn_workgroup_id_x());

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockIdx_t);
};

extern "C" __device__ uint64_t __ockl_get_local_size(uint32_t);
struct __cuda_builtin_blockDim_t {
  __CUDA_DEVICE_BUILTIN(x, __ockl_get_local_size(0));
  __CUDA_DEVICE_BUILTIN(y, __ockl_get_local_size(1));
  __CUDA_DEVICE_BUILTIN(z, __ockl_get_local_size(2));

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockDim_t);
};

extern "C" __device__ uint64_t __ockl_get_num_groups(uint32_t);
struct __cuda_builtin_gridDim_t {
  __CUDA_DEVICE_BUILTIN(x, __ockl_get_num_groups(0));
  __CUDA_DEVICE_BUILTIN(y, __ockl_get_num_groups(1));
  __CUDA_DEVICE_BUILTIN(z, __ockl_get_num_groups(2));

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
