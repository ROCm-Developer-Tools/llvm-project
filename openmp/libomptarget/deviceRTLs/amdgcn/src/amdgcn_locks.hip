//===-- amdgcn_locks.hip - AMDGCN OpenMP GPU lock implementation -- HIP -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of openmp lock functions
// Warning: Derived from nvptx, suspected to deadlock on amdgcn: aomp/issues/50
//
//===----------------------------------------------------------------------===//

#include "common/debug.h"
#include "common/target_atomic.h"
#include "target_impl.h"

#define __OMP_SPIN 1000
#define UNSET 0u
#define SET 1u

DEVICE void __kmpc_impl_init_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
}

DEVICE void __kmpc_impl_destroy_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
}

DEVICE void __kmpc_impl_set_lock(omp_lock_t *lock) {
  // int atomicCAS(int* address, int compare, int val);
  // (old == compare ? val : old)

  // TODO: not sure spinning is a good idea here..
  while (__kmpc_atomic_cas(lock, UNSET, SET) != UNSET) {
    uint64_t start = __clock64();
    uint64_t now;
    for (;;) {
      now = __clock64();
      uint64_t cycles = now > start ? now - start : now + (0xffffffff - start);
      if (cycles >= __OMP_SPIN * GetBlockIdInKernel()) {
        break;
      }
    }
  } // wait for 0 to be the read value
}

DEVICE void __kmpc_impl_unset_lock(omp_lock_t *lock) {
  (void)__kmpc_atomic_exchange(lock, UNSET);
}

DEVICE int __kmpc_impl_test_lock(omp_lock_t *lock) {
  // int atomicCAS(int* address, int compare, int val);
  // (old == compare ? val : old)
  return __kmpc_atomic_add(lock, 0u);
}
