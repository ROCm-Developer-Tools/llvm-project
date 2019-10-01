//===------------ sync.h - NVPTX OpenMP synchronizations --------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Include all synchronization.
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"
#include "target_impl.h"

////////////////////////////////////////////////////////////////////////////////
// KMP Ordered calls
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_ordered(kmp_Ident *loc, int32_t tid) {
  PRINT0(LD_IO, "call kmpc_ordered\n");
}

EXTERN void __kmpc_end_ordered(kmp_Ident *loc, int32_t tid) {
  PRINT0(LD_IO, "call kmpc_end_ordered\n");
}

////////////////////////////////////////////////////////////////////////////////
// KMP Barriers
////////////////////////////////////////////////////////////////////////////////

// a team is a block: we can use CUDA native synchronization mechanism
// FIXME: what if not all threads (warps) participate to the barrier?
// We may need to implement it differently

EXTERN int32_t __kmpc_cancel_barrier(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_IO, "call kmpc_cancel_barrier\n");
  __kmpc_barrier(loc_ref, tid);
  PRINT0(LD_SYNC, "completed kmpc_cancel_barrier\n");
  return 0;
}

EXTERN void __kmpc_barrier(kmp_Ident *loc_ref, int32_t tid) {
  if (checkRuntimeUninitialized(loc_ref)) {
    ASSERT0(LT_FUSSY, checkSPMDMode(loc_ref),
            "Expected SPMD mode with uninitialized runtime.");
    __kmpc_barrier_simple_spmd(loc_ref, tid);
  } else {
    tid = GetLogicalThreadIdInBlock(checkSPMDMode(loc_ref));
    int numberOfActiveOMPThreads =
        GetNumberOfOmpThreads(checkSPMDMode(loc_ref));
    if (numberOfActiveOMPThreads > 1) {
      if (checkSPMDMode(loc_ref)) {
        __kmpc_barrier_simple_spmd(loc_ref, tid);
      } else {
        // The #threads parameter must be rounded up to the WARPSIZE.
        int threads =
            WARPSIZE * ((numberOfActiveOMPThreads + WARPSIZE - 1) / WARPSIZE);

        PRINT(LD_SYNC,
              "call kmpc_barrier with %d omp threads, sync parameter %d\n",
              (int)numberOfActiveOMPThreads, (int)threads);
        // Barrier #1 is for synchronization among active threads.
        named_sync(L1_BARRIER, threads);
      }
    } // numberOfActiveOMPThreads > 1
    PRINT0(LD_SYNC, "completed kmpc_barrier\n");
  }
}

// Emit a simple barrier call in SPMD mode.  Assumes the caller is in an L0
// parallel region and that all worker threads participate.
EXTERN void __kmpc_barrier_simple_spmd(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_barrier_simple_spmd\n");
  __kmpc_impl_syncthreads();
  PRINT0(LD_SYNC, "completed kmpc_barrier_simple_spmd\n");
}

///////////////////////////////////////////////////////////////////////////////
/// The following four functions are used to coordinate barriers between
/// the master and worker warps in a generic kernel.  The amdgcn architecture
/// does not have a partial barrier so this coordination needs to be
/// recover if a user or implicit barrier is encountered in the sequential
/// part of the master warp or the parallel part of the worker warps.
///
/// __kmpc_amd_worker_start has a barrier to prevent worker warps
/// from starting till the master warp sets the workFn.  This function sets
/// workers_active to true after the barrier to keep the master warp
/// at its barrier in case a worker encounters an explicit or implicit
/// barrier.
EXTERN void __kmpc_amd_worker_start(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_amd_worker_start\n");
  omptarget_workers_active = false;
  __kmpc_impl_syncthreads();
  while(omptarget_master_active) __kmpc_impl_syncthreads();
  omptarget_workers_active = true;
  PRINT0(LD_SYNC, "completed kmpc_amd_worker_start\n");
}

/// __kmpc_amd_worker_end sets workers_active to false and then
/// issues a barrier to release the master warp to terminate or get
/// the next subregion of work to process.
EXTERN void __kmpc_amd_worker_end(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_amd_worker_end\n");
  omptarget_workers_active = false;
  __kmpc_impl_syncthreads(); // to sync with 2nd barrier in master_end
  PRINT0(LD_SYNC, "completed kmpc_amd_worker_end\n");
}

/// __kmpc_amd_master_start is executed first by the master.  It sets
/// master_active to true to keep the worker warps at its first barrier
/// in case the sequential part of the target region encounters any
/// barrier, explicit or otherwise.
EXTERN void __kmpc_amd_master_start(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_amd_master_start\n");
  omptarget_master_active = true;
  PRINT0(LD_SYNC, "completed kmpc_amd_master_start\n");
}

/// __kmpc_amd_master_end sets master_active to false and then enters
/// the double barrier. The first releases the worker warps. The 2nd
/// barrier holds the master warp until the workers are done. If the worker
/// warps encounter a user barrier (implicitly or explicitly), the master
/// warp needs to loop at the barrier until it knows the worker is
/// really done by testing omptarget_workers_active.
EXTERN void __kmpc_amd_master_end(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_amd_master_end\n");
  omptarget_master_active = false;
  __kmpc_impl_syncthreads();
  __kmpc_impl_syncthreads();
  while(omptarget_workers_active) __kmpc_impl_syncthreads();
  PRINT0(LD_SYNC, "completed kmpc_amd_master_end\n");
}

// Emit a simple barrier call in Generic mode.  Assumes the caller is in an L0
// parallel region and that all worker threads participate.
EXTERN void __kmpc_barrier_simple_generic(kmp_Ident *loc_ref, int32_t tid) {
  int numberOfActiveOMPThreads = GetNumberOfThreadsInBlock() - WARPSIZE;
  // The #threads parameter must be rounded up to the WARPSIZE.
  int threads =
      WARPSIZE * ((numberOfActiveOMPThreads + WARPSIZE - 1) / WARPSIZE);

  PRINT(LD_SYNC,
        "call kmpc_barrier_simple_generic with %d omp threads, sync parameter "
        "%d\n",
        (int)numberOfActiveOMPThreads, (int)threads);
  // Barrier #1 is for synchronization among active threads.
  named_sync(L1_BARRIER, threads);
  PRINT0(LD_SYNC, "completed kmpc_barrier_simple_generic\n");
}

////////////////////////////////////////////////////////////////////////////////
// KMP MASTER
////////////////////////////////////////////////////////////////////////////////

EXTERN int32_t __kmpc_master(kmp_Ident *loc, int32_t global_tid) {
  PRINT0(LD_IO, "call kmpc_master\n");
  return IsTeamMaster(global_tid);
}

EXTERN void __kmpc_end_master(kmp_Ident *loc, int32_t global_tid) {
  PRINT0(LD_IO, "call kmpc_end_master\n");
  ASSERT0(LT_FUSSY, IsTeamMaster(global_tid), "expected only master here");
}

////////////////////////////////////////////////////////////////////////////////
// KMP SINGLE
////////////////////////////////////////////////////////////////////////////////

EXTERN int32_t __kmpc_single(kmp_Ident *loc, int32_t global_tid) {
  PRINT0(LD_IO, "call kmpc_single\n");
  // decide to implement single with master; master get the single
  return IsTeamMaster(global_tid);
}

EXTERN void __kmpc_end_single(kmp_Ident *loc, int32_t global_tid) {
  PRINT0(LD_IO, "call kmpc_end_single\n");
  // decide to implement single with master: master get the single
  ASSERT0(LT_FUSSY, IsTeamMaster(global_tid), "expected only master here");
  // sync barrier is explicitely called... so that is not a problem
}

////////////////////////////////////////////////////////////////////////////////
// Flush
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_flush(kmp_Ident *loc) {
  PRINT0(LD_IO, "call kmpc_flush\n");
  __threadfence_system();
}

////////////////////////////////////////////////////////////////////////////////
// Vote
////////////////////////////////////////////////////////////////////////////////

#ifdef __AMDGCN__
EXTERN int64_t __kmpc_warp_active_thread_mask64() {
  PRINT0(LD_IO, "call __kmpc_warp_active_thread_mask64\n");
  return __kmpc_impl_activemask();
}
#else
EXTERN int32_t __kmpc_warp_active_thread_mask() {
  PRINT0(LD_IO, "call __kmpc_warp_active_thread_mask\n");
  return __ACTIVEMASK();
}
#endif
