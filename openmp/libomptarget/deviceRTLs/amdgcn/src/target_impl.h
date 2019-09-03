//===------------ target_impl.h - NVPTX OpenMP GPU options ------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of target specific functions
//
//===----------------------------------------------------------------------===//
#ifndef _TARGET_IMPL_H_
#define _TARGET_IMPL_H_

#include <stdint.h>

#include "omptarget-nvptx.h"
#include "option.h"

INLINE void __kmpc_impl_unpack(uint64_t val, uint32_t &lo, uint32_t &hi) {
  lo = (uint32_t)(val & 0x00000000FFFFFFFFL);
  hi = (uint32_t)((val & 0xFFFFFFFF00000000L) >> 32);
}

INLINE uint64_t __kmpc_impl_pack(uint32_t lo, uint32_t hi) {
  return (((uint64_t)hi) << 32) | (uint64_t)lo;
}

typedef uint64_t __kmpc_impl_lanemask_t;

INLINE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  return __lanemask_lt();
}

INLINE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  return __lanemask_gt();
}

INLINE uint64_t __kmpc_impl_ffs(uint64_t x) { return __ffsll(x); }

INLINE uint64_t __kmpc_impl_popc(uint64_t x) { return __popcll(x); }

INLINE __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
  return __ballot64(1);
}

INLINE int32_t __kmpc_impl_shfl_sync(__kmpc_impl_lanemask_t Mask, int32_t Var,
                                     int32_t SrcLane) {
  return __SHFL_SYNC(Mask, Var, SrcLane);
}

INLINE int32_t __kmpc_impl_shfl_down_sync(__kmpc_impl_lanemask_t Mask,
                                          int32_t Var, uint32_t Delta,
                                          int32_t Width) {
  return __SHFL_DOWN_SYNC(Mask, Var, Delta, Width);
}

INLINE void __kmpc_impl_syncthreads() { __SYNCTHREADS(); }

#endif
