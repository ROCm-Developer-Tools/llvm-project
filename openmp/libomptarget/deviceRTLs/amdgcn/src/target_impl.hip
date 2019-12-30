//===------------ target_impl.hip - AMDGCN OpenMP GPU options ------ HIP -*-===//
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

// warp vote function
EXTERN uint64_t __ballot64(int predicate);
// initialized with a 64-bit mask with bits set in positions less than the
// thread's lane number in the warp
EXTERN uint64_t __lanemask_lt();
// initialized with a 64-bit mask with bits set in positions greater than the
// thread's lane number in the warp
EXTERN uint64_t __lanemask_gt();

// CU id
EXTERN unsigned __smid();

DEVICE __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
  return __ballot64(1);
}

DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  return __lanemask_lt();
}

DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  return __lanemask_gt();
}

DEVICE uint32_t __kmpc_impl_smid() {
  return __smid();
}
