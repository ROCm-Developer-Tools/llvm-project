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

__device__ static uint64_t __clock64(void) {
#if __AMDGCN__ > 800
  return __builtin_amdgcn_s_memrealtime();
#else
  __device__ uint64_t __llvm_amdgcn_s_memrealtime(void);
  return __llvm_amdgcn_s_memrealtime();
#endif
}

#endif
