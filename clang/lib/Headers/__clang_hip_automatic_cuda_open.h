/*===---- __clang_hip_automatic_cuda_open.h - CUDA runtime support --------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

/*
 * WARNING: This header is intended to be directly -include'd by
 * the compiler and is not supposed to be included by users.
 * This file is automatically included in HIP automatic mode.
 * when the default is overriden with --hip-auto-headers=cuda_open.
 *
 */

#ifndef __CLANG_HIP_AUTOMATIC_CUDA_OPEN_H__
#define __CLANG_HIP_AUTOMATIC_CUDA_OPEN_H__

#if defined(__HIP__) && defined(__clang__)

#define __CUDA__

#define CUDA_VERSION 9000

#define __USE_OPEN_HEADERS__

// Include some forward declares that must come before cmath.
#include <cuda_open/__clang_cuda_open_math_forward_declares.h>

// Include some standard headers to avoid CUDA headers including them
// while some required macros (like __THROW) are in a weird state.
#include <cmath>
#include <cstdlib>
#include <limits.h>
#include <stdlib.h>

// Preserve common macros that will be changed below by us or by CUDA
// headers.
#pragma push_macro("__THROW")
#pragma push_macro("__CUDA_ARCH__")

#include <cuda_open/cuda_open.h>

#include <cuda_open/__clang_cuda_open_builtin_vars.h>
#include <cuda_open/__clang_cuda_open_nv_declares.h>
#include <cuda_open/__clang_cuda_open_device_functions.h>
#include <cuda_open/__clang_cuda_open_cmath.h>

// No need for device_launch_parameters.h as __clang_cuda_builtin_vars.h above
// has taken care of builtin variables declared in the file.
#define __DEVICE_LAUNCH_PARAMETERS_H__

// {math,device}_functions.h only have declarations of the
// functions. We don't need them as we're going to pull in their
// definitions from .hpp files.
#define __DEVICE_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__
#define __COMMON_FUNCTIONS_H__
// device_functions_decls is replaced by __clang_cuda_device_functions.h
// included below.
#define __DEVICE_FUNCTIONS_DECLS_H__

#undef __CUDACC__
#if CUDA_VERSION < 9000
#define __CUDABE__
#else
#define __CUDA_LIBDEVICE__
#endif

// Disables definitions of device-side runtime support stubs in
// cuda_device_runtime_api.h
#include <cuda_open/driver_types.h>
#include <cuda_open/host_config.h>
#include <cuda_open/host_defines.h>

#undef __CUDABE__
#undef __CUDA_LIBDEVICE__
#define __CUDACC__
#include <cuda_open/cuda_open_runtime.h>

#undef __CUDACC__
#define __CUDABE__

// CUDA headers use __nvvm_memcpy and __nvvm_memset which Clang does
// not have at the moment. Emulate them with a builtin memcpy/memset.
#define __nvvm_memcpy(s, d, n, a) __builtin_memcpy(s, d, n)
#define __nvvm_memset(d, c, n, a) __builtin_memset(d, c, n)

#include <cuda_open/device_runtime.h>
// device_runtime.h defines __cxa_* macros that will conflict with
// cxxabi.h.
// FIXME: redefine these as __device__ functions.
#undef __cxa_vec_ctor
#undef __cxa_vec_cctor
#undef __cxa_vec_dtor
#undef __cxa_vec_new
#undef __cxa_vec_new2
#undef __cxa_vec_new3
#undef __cxa_vec_delete2
#undef __cxa_vec_delete
#undef __cxa_vec_delete3
#undef __cxa_pure_virtual

// math_functions.hpp expects this host function be defined on MacOS, but it
// ends up not being there because of the games we play here.  Just define it
// ourselves; it's simple enough.
#ifdef __APPLE__
inline __host__ double __signbitd(double x) { return std::signbit(x); }
#endif

// __THROW is redefined to be empty by device_functions_decls.h in CUDA. Clang's
// counterpart does not do it, so we need to make it empty here to keep
// following CUDA includes happy.
#undef __THROW
#define __THROW

// CUDA 8.0.41 relies on __USE_FAST_MATH__ and __CUDA_PREC_DIV's values.
// Previous versions used to check whether they are defined or not.
// CU_DEVICE_INVALID macro is only defined in 8.0.41, so we use it
// here to detect the switch.

#if defined(CU_DEVICE_INVALID)
#if !defined(__USE_FAST_MATH__)
#define __USE_FAST_MATH__ 0
#endif

#if !defined(__CUDA_PREC_DIV)
#define __CUDA_PREC_DIV 0
#endif
#endif

// Temporarily poison __host__ macro to ensure it's not used by any of
// the headers we're about to include.
#pragma push_macro("__host__")
#define __host__ UNEXPECTED_HOST_ATTRIBUTE

// device_functions.hpp and math_functions*.hpp use 'static
// __forceinline__' (with no __device__) for definitions of device
// functions. Temporarily redefine __forceinline__ to include
// __device__.
#pragma push_macro("__forceinline__")
#define __forceinline__ __device__ __inline__ __attribute__((always_inline))
#if CUDA_VERSION < 9000
#include <cuda_open/device_functions.hpp>
#endif

// math_function.hpp uses the __USE_FAST_MATH__ macro to determine whether we
// get the slow-but-accurate or fast-but-inaccurate versions of functions like
// sin and exp.  This is controlled in clang by -fcuda-approx-transcendentals.
//
// device_functions.hpp uses __USE_FAST_MATH__ for a different purpose (fast vs.
// slow divides), so we need to scope our define carefully here.
#pragma push_macro("__USE_FAST_MATH__")
#if defined(__CLANG_CUDA_APPROX_TRANSCENDENTALS__)
#define __USE_FAST_MATH__ 1
#endif

#include <cuda_open/math_functions.hpp>

#pragma pop_macro("__USE_FAST_MATH__")

#if CUDA_VERSION < 9000
#include <cuda_open/math_functions_dbl.hpp>
#endif
#pragma pop_macro("__forceinline__")

// Pull in host-only functions that are only available when neither
// __CUDACC__ nor __CUDABE__ are defined.
#undef __MATH_FUNCTIONS_HPP__
#undef __CUDABE__

// Now include *.hpp with definitions of various GPU functions.  Alas,
// a lot of thins get declared/defined with __host__ attribute which
// we don't want and we have to define it out. We also have to include
// {device,math}_functions.hpp again in order to extract the other
// branch of #if/else inside.
#ifdef __host__
#undef __host__
#endif
#define __host__
#undef __CUDABE__
#define __CUDACC__
#include <cuda_open/device_atomic_functions.hpp>
#include <cuda_open/device_functions.hpp>
#define __CUDABE__
#include <cuda_open/device_double_functions.h>
#undef __CUDABE__

#undef __MATH_FUNCTIONS_HPP__

// math_functions.hpp defines ::signbit as a __host__ __device__ function.  This
// conflicts with libstdc++'s constexpr ::signbit, so we have to rename
// math_function.hpp's ::signbit.  It's guarded by #undef signbit, but that's
// conditional on __GNUC__.  :)
#pragma push_macro("signbit")
#pragma push_macro("__GNUC__")
#undef __GNUC__
#define signbit __ignored_cuda_signbit

// CUDA-9 omits device-side definitions of some math functions if it sees
// include guard from math.h wrapper from libstdc++. We have to undo the header
// guard temporarily to get the definitions we need.
#pragma push_macro("_GLIBCXX_MATH_H")
#pragma push_macro("_LIBCPP_VERSION")
#if CUDA_VERSION >= 9000
#undef _GLIBCXX_MATH_H
// We also need to undo another guard that checks for libc++ 3.8+
#ifdef _LIBCPP_VERSION
#define _LIBCPP_VERSION 3700
#endif
#endif

#include <cuda_open/math_functions.hpp>
#pragma pop_macro("_GLIBCXX_MATH_H")
#pragma pop_macro("_LIBCPP_VERSION")
#pragma pop_macro("__GNUC__")
#pragma pop_macro("signbit")

#pragma pop_macro("__host__")

#include <cuda_open/texture_indirect_functions.h>

// Restore state of __CUDA_ARCH__ and __THROW we had on entry.
#pragma pop_macro("__CUDA_ARCH__")
#pragma pop_macro("__THROW")

// Set up compiler macros expected to be seen during compilation.
#undef __CUDABE__
#define __CUDACC__

extern "C" {
// Device-side CUDA system calls.
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls
// We need these declarations and wrappers for device-side
// malloc/free/printf calls to work without relying on
// -fcuda-disable-target-call-checks option.
__device__ int vprintf(const char *, const char *);
__device__ void free(void *) __attribute((nothrow));
__device__ void *malloc(size_t) __attribute((nothrow)) __attribute__((malloc));
__device__ void __assertfail(const char *__message, const char *__file,
                             unsigned __line, const char *__function,
                             size_t __charSize) __attribute__((noreturn));

// In order for standard assert() macro on linux to work we need to
// provide device-side __assert_fail()
__device__ static inline void __assert_fail(const char *__message,
                                            const char *__file, unsigned __line,
                                            const char *__function) {
  __assertfail(__message, __file, __line, __function, sizeof(char));
}

// Clang will convert printf into vprintf, but we still need
// device-side declaration for it.
__device__ int printf(const char *, ...);
} // extern "C"

// We also need device-side std::malloc and std::free.
namespace std {
__device__ static inline void free(void *__ptr) { ::free(__ptr); }
__device__ static inline void *malloc(size_t __size) {
  return ::malloc(__size);
}
} // namespace std

// Out-of-line implementations from __clang_cuda_builtin_vars.h.  These need to
// come after we've pulled in the definition of uint3 and dim3.

__device__ inline __cuda_builtin_threadIdx_t::operator uint3() const {
  uint3 ret;
  ret.x = x;
  ret.y = y;
  ret.z = z;
  return ret;
}

__device__ inline __cuda_builtin_blockIdx_t::operator uint3() const {
  uint3 ret;
  ret.x = x;
  ret.y = y;
  ret.z = z;
  return ret;
}

__device__ inline __cuda_builtin_blockDim_t::operator dim3() const {
  return dim3(x, y, z);
}

__device__ inline __cuda_builtin_gridDim_t::operator dim3() const {
  return dim3(x, y, z);
}

#include <cuda_open/__clang_cuda_open_intrinsics.h>

// Eventually open headers may have defines with asm(ptx) so we
// still many need these overrides
//#include <__clang_cuda_asmptx_overrides.h>

#include "hip/hip_host_runtime_api.h"
#ifdef __HIP_DEVICE_COMPILE__
__device__ void __syncthreads();
//__device__ static inline void __threadfence_block(void);
//__device__ unsigned int __popcll(unsigned long long int x);
//  Overloaded min
__device__ float min(float x, float y);
__device__ int min(int x, int y);
__device__ double min(double x, double y);
__device__ long long min(long long x, long long y);
#endif

#endif // __HIP__ && clang
#endif // __CLANG_HIP_AUTOMATIC_CUDA_OPEN_H__
