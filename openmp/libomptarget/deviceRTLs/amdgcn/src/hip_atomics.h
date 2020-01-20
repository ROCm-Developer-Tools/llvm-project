//===---- hip_atomics.h - Declarations of hip atomic functions ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_AMDGCN_HIP_ATOMICS_H
#define OMPTARGET_AMDGCN_HIP_ATOMICS_H

#include "target_impl.h"

namespace {

template <typename T> DEVICE T atomicAdd(volatile T *x, T v) {
  return __atomic_fetch_add(x, v, __ATOMIC_SEQ_CST);
}

// Only implemented for i32 as that's the only call site
EXTERN uint32_t __amdgcn_atomic_inc_i32(uint32_t *, uint32_t);
INLINE uint32_t atomicInc(uint32_t *address, uint32_t val) {
  return __amdgcn_atomic_inc_i32(address, val);
}

template <typename T> DEVICE T atomicMax(volatile T *address, T val) {
  return __opencl_atomic_fetch_max((_Atomic volatile T *)address, val,
                                   __ATOMIC_SEQ_CST,
                                   __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
}

template <typename T> DEVICE T atomicExch(volatile T *address, T val) {
  return __opencl_atomic_exchange((_Atomic volatile T *)address, val,
                                  __ATOMIC_SEQ_CST,
                                  __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
}

template <typename T>
DEVICE T atomicCAS(volatile T *address, T compare, T val) {
  (void)__opencl_atomic_compare_exchange_strong(
      (_Atomic volatile T *)address, &compare, val, __ATOMIC_SEQ_CST,
      __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
  return compare;
}

} // namespace
#endif
