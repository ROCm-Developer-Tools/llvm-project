//===------------ target_impl.h - AMDGCN OpenMP GPU options ------ CUDA -*-===//
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
#ifndef __AMDGCN__
// The amdgcn deviceRTL can be compiled for nvptx. Including the corresponding
// target_impl file (which sets the same header guard) maintains this property
// while allowing use of the __kmpc_impl constructs in the nvptx subsections.
// This include can be dropped once sources are deduplicated to a common subdir
#include "../../nvptx/src/target_impl.h"
#endif

#ifndef _TARGET_IMPL_H_
#define _TARGET_IMPL_H_

#include <stdint.h>

#include "option.h"

#ifndef __AMDGCN__
#error "amdgcn target_impl.h expects to be compiled under __AMDGCN__"
#endif

// warp vote function
EXTERN uint64_t __ballot64(int predicate);
// initialized with a 64-bit mask with bits set in positions less than the
// thread's lane number in the warp
EXTERN uint64_t __lanemask_lt();
// initialized with a 64-bit mask with bits set in positions greater than the
// thread's lane number in the warp
EXTERN uint64_t __lanemask_gt();

EXTERN void llvm_amdgcn_s_barrier();

INLINE void __kmpc_impl_unpack(uint64_t val, uint32_t &lo, uint32_t &hi) {
  lo = (uint32_t)(val & 0x00000000FFFFFFFFL);
  hi = (uint32_t)((val & 0xFFFFFFFF00000000L) >> 32);
}

INLINE uint64_t __kmpc_impl_pack(uint32_t lo, uint32_t hi) {
  return (((uint64_t)hi) << 32) | (uint64_t)lo;
}

static const __kmpc_impl_lanemask_t __kmpc_impl_all_lanes =
    UINT64_C(0xffffffffffffffff);

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

INLINE int32_t __kmpc_impl_shfl_sync(__kmpc_impl_lanemask_t, int32_t Var,
                                     int32_t SrcLane) {
  return __shfl(Var, SrcLane, WARPSIZE);
}

INLINE int32_t __kmpc_impl_shfl_down_sync(__kmpc_impl_lanemask_t, int32_t Var,
                                          uint32_t Delta, int32_t Width) {
  return __shfl_down(Var, Delta, Width);
}

INLINE void __kmpc_impl_syncthreads() { llvm_amdgcn_s_barrier(); }

#endif
