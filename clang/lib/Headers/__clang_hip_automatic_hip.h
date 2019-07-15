/*===---- __clang_hip_automatic_hip.h - CUDA runtime support --------===
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
 */

#ifndef __CLANG_HIP_AUTOMATIC_HIP_H__
#define __CLANG_HIP_AUTOMATIC_HIP_H__

#if defined(__HIP__) && defined(__clang__)

#ifdef __HIP_DEVICE_COMPILE__
#define __align__(n) __attribute__((aligned(n)))
#else
#define __align__(n) __declspec(align(n))
#endif

#include <hip/hip_runtime.h>

extern "C" __device__ int printf(const char *, ...);

#endif // __HIP__ && clang
#endif // __CLANG_HIP_AUTOMATIC_HIP_H__
