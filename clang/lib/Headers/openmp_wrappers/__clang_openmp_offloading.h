/*===- __clang_openmp_offloading.h - auto-include for omp offloading  ------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_OPENMP_OFFLOADING_H__
#define __CLANG_OPENMP_OFFLOADING_H__

#ifndef _OPENMP
#error "This file is for OpenMP compilation only."
#endif

// Insert requires unified_shared_memory directive with option -offload_usm
// for both host and device passes.
#ifdef _OPENMP_USM
#pragma omp requires unified_shared_memory
#endif

// On device pass, include __clang_openmp_device_functions.h
#if defined(__AMDGCN__) || defined(__NVPTX__)
#include "openmp_wrappers/__clang_openmp_device_functions.h"
#endif

#endif // __CLANG_OPENMP_OFFLOADING_H__
