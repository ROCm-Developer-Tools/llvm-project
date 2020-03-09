//===---- amdgcn_reduction.hip - extensions to OpenMP reduction --- HIP -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains amdgcn extensions to the the implementation of reduction
//
//===----------------------------------------------------------------------===//

#include "common/omptarget.h"
#include "common/target_atomic.h"
#include "target_impl.h"

// may eventually remove this
EXTERN
int32_t __gpu_block_reduce() {
  bool isSPMDExecutionMode = isSPMDMode();
  int nt = GetNumberOfOmpThreads(isSPMDExecutionMode);
  if (nt != GetNumberOfThreadsInBlock())
    return 0;

  __kmpc_impl_lanemask_t tnum = __kmpc_impl_activemask();
#ifdef __AMDGCN__
  if (tnum == __kmpc_impl_all_lanes) {
    return 0;
  }
#else
   if (tnum != (~0x0)) { // assume swapSize is 32
     return 0;
   }
#endif

  return 1;
}

EXTERN
int32_t __kmpc_reduce_gpu(kmp_Ident *loc, int32_t global_tid, int32_t num_vars,
                          size_t reduce_size, void *reduce_data,
                          void *reduce_array_size, kmp_ReductFctPtr *reductFct,
                          kmp_CriticalName *lck) {
  int threadId = GetLogicalThreadIdInBlock(checkSPMDMode(loc));
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  int numthread;
  if (currTaskDescr->IsParallelConstruct()) {
    numthread = GetNumberOfOmpThreads(checkSPMDMode(loc));
  } else {
    numthread = GetNumberOfOmpTeams();
  }

  if (numthread == 1)
    return 1;
  if (!__gpu_block_reduce())
    return 2;
  if (GetThreadIdInBlock() == 0)
    return 1;
  return 0;
}

EXTERN
int32_t __kmpc_reduce_combined(kmp_Ident *loc) {
  return GetThreadIdInBlock() == 0 ? 2 : 0;
}

EXTERN
int32_t __kmpc_reduce_simd(kmp_Ident *loc) {
  return (GetThreadIdInBlock() % WARPSIZE == 0) ? 1 : 0;
}

