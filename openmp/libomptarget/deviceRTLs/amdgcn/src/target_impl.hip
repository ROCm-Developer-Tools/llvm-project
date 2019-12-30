//===------------ target_impl.hip - AMDGCN OpenMP GPU options ----- HIP -*-===//
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

#include "target_impl.h"

// Implementations initially derived from hip/hcc_detail/device_functions.h

// warp vote function
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
  // 33 is ICMP_NE from llvm/include/llvm/IR/InstrTypes.h
  return __builtin_amdgcn_uicmp(1, 0, 33);
}

EXTERN uint32_t __ockl_lane_u32(void);

// initialized with a 64-bit mask with bits set in positions less than the
// thread's lane number in the warp
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  uint32_t lane = __ockl_lane_u32();
  int64_t ballot = __kmpc_impl_activemask();
  uint64_t mask = ((uint64_t)1 << lane) - (uint64_t)1;
  return mask & ballot;
}

// initialized with a 64-bit mask with bits set in positions greater than the
// thread's lane number in the warp
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  uint32_t lane = __ockl_lane_u32();
  if (lane == 63)
    return 0;
  uint64_t ballot = __kmpc_impl_activemask();
  uint64_t mask = (~((uint64_t)0)) << (lane + 1);
  return mask & ballot;
}

// CU id
EXTERN unsigned __smid();
DEVICE uint32_t __kmpc_impl_smid() { return __smid(); }
