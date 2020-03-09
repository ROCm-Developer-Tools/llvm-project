//===--- amdgcn_interface.h - OpenMP interface definitions ------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _AMDGCN_INTERFACE_H_
#define _AMDGCN_INTERFACE_H_

#include <stdint.h>
#include <stddef.h>

#define EXTERN extern "C" __attribute__((device))
typedef uint64_t __kmpc_impl_lanemask_t;
typedef uint32_t omp_lock_t; /* arbitrary type of the right length */

////////////////////////////////////////////////////////////////////////////////
// OpenMP interface
////////////////////////////////////////////////////////////////////////////////

EXTERN int omp_get_device_num(void);
EXTERN int omp_ext_get_warp_id(void);
EXTERN int omp_ext_get_lane_id(void);
EXTERN int omp_ext_get_master_thread_id(void);
EXTERN int omp_ext_get_smid(void);
EXTERN int omp_ext_is_spmd_mode(void);
EXTERN unsigned long long omp_ext_get_active_threads_mask(void);

////////////////////////////////////////////////////////////////////////////////
// kmp specifc types
////////////////////////////////////////////////////////////////////////////////

// task dep defs
#define KMP_TASKDEP_IN 0x1u
#define KMP_TASKDEP_OUT 0x2u
typedef struct kmp_TaskDep_Public {
  void *addr;
  size_t len;
  uint8_t flags; // bit 0: in, bit 1: out
} kmp_TaskDep_Public;

// flags that interpret the interface part of tasking flags
#define KMP_TASK_IS_TIED 0x1
#define KMP_TASK_FINAL 0x2
#define KMP_TASK_MERGED_IF0 0x4 /* unused */
#define KMP_TASK_DESTRUCTOR_THUNK 0x8

// flags for task setup return
#define KMP_CURRENT_TASK_NOT_SUSPENDED 0
#define KMP_CURRENT_TASK_SUSPENDED 1

////////////////////////////////////////////////////////////////////////////////
// flags for kstate (all bits initially off)
////////////////////////////////////////////////////////////////////////////////

// first 2 bits used by kmp_Reduction (defined in kmp_reduction.cpp)
#define KMP_REDUCTION_MASK 0x3
#define KMP_SKIP_NEXT_CALL 0x4
#define KMP_SKIP_NEXT_CANCEL_BARRIER 0x8

////////////////////////////////////////////////////////////////////////////////
// external interface
////////////////////////////////////////////////////////////////////////////////

typedef struct ident ident_t;
typedef ident_t kmp_Ident;

// query
EXTERN int32_t __kmpc_global_num_threads(kmp_Ident *loc); // missing
EXTERN int32_t __kmpc_bound_thread_num(kmp_Ident *loc);   // missing
EXTERN int32_t __kmpc_bound_num_threads(kmp_Ident *loc);  // missing
EXTERN int32_t __kmpc_in_parallel(kmp_Ident *loc);        // missing

// sync barrier
EXTERN void __kmpc_amd_worker_start(kmp_Ident *loc_ref, int32_t tid);
EXTERN void __kmpc_amd_worker_end(kmp_Ident *loc_ref, int32_t tid);
EXTERN void __kmpc_amd_master_start(kmp_Ident *loc_ref, int32_t tid);
EXTERN void __kmpc_amd_master_end(kmp_Ident *loc_ref, int32_t tid);
EXTERN void __kmpc_amd_master_terminate(kmp_Ident *loc_ref, int32_t tid);

#endif
