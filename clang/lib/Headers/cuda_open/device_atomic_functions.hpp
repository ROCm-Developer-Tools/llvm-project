//===--------- cuda_open/device_atomic_functions.hpp  ---------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef __CUDA_OPEN_DEVICE_ATOMIC_FUNCTIONS_HPP__
#define __CUDA_OPEN_DEVICE_ATOMIC_FUNCTIONS_HPP__
#ifndef __CUDA__
#error "This file is for CUDA compilation only."
#endif
#ifndef __USE_OPEN_HEADERS__
#error "This file requires -D__USE_OPEN_HEADERS__"
#endif

#define __NOOVL__ extern "C" __attribute__((device, always_inline)) const
#ifdef __cplusplus
extern "C" {
#endif

#define __OVERL__ __attribute__((device, always_inline, overloadable)) const

// atomicAdd()
__OVERL__ unsigned int atomicAdd(unsigned int *address, unsigned int val);
__OVERL__ int atomicAdd(int *address, int val);
__OVERL__ float atomicAdd(float *address, float val);
__OVERL__ unsigned long long int atomicAdd(unsigned long long int *address,
                                           unsigned long long int val);

// atomicCAS()
__OVERL__ unsigned int atomicCAS(unsigned int *address, unsigned int compare,
                                 unsigned int val);
__OVERL__ int atomicCAS(int *address, int compare, int val);
__OVERL__ unsigned long long int atomicCAS(unsigned long long int *address,
                                           unsigned long long int compare,
                                           unsigned long long int val);

// atomicSub()
__OVERL__ int atomicSub(int *address, int val);
__OVERL__ unsigned int atomicSub(unsigned int *address, unsigned int val);

// atomicExch()
__OVERL__ int atomicExch(int *address, int val);
__OVERL__ unsigned int atomicExch(unsigned int *address, unsigned int val);
__OVERL__ unsigned long long int atomicExch(unsigned long long int *address,
                                            unsigned long long int val);
__OVERL__ float atomicExch(float *address, float val);

// atomicMin()
__OVERL__ int atomicMin(int *address, int val);
__OVERL__ unsigned int atomicMin(unsigned int *address, unsigned int val);
__OVERL__ unsigned long long int atomicMin(unsigned long long int *address,
                                           unsigned long long int val);

// atomicMax()
__OVERL__ int atomicMax(int *address, int val);
__OVERL__ unsigned int atomicMax(unsigned int *address, unsigned int val);
__OVERL__ unsigned long long int atomicMax(unsigned long long int *address,
                                           unsigned long long int val);
// atomicAnd()
__OVERL__ int atomicAnd(int *address, int val);
__OVERL__ unsigned int atomicAnd(unsigned int *address, unsigned int val);
__OVERL__ unsigned long long int atomicAnd(unsigned long long int *address,
                                           unsigned long long int val);

// atomicOr()
__OVERL__ int atomicOr(int *address, int val);
__OVERL__ unsigned int atomicOr(unsigned int *address, unsigned int val);
__OVERL__ unsigned long long int atomicOr(unsigned long long int *address,
                                          unsigned long long int val);

// atomicXor()
__OVERL__ int atomicXor(int *address, int val);
__OVERL__ unsigned int atomicXor(unsigned int *address, unsigned int val);
__OVERL__ unsigned long long int atomicXor(unsigned long long int *address,
                                           unsigned long long int val);

// atomicInc()
__OVERL__ unsigned int atomicInc(unsigned int *address, unsigned int val);

// atomicDec()
__OVERL__ unsigned int atomicDec(unsigned int *address, unsigned int val);

// -------------------------------------
// These are the non overloaded functions used to implement the
// above overloaded atomic functions
__NOOVL__ unsigned atomic_exchange_unsigned(unsigned *addr, unsigned val);
__NOOVL__ unsigned atomic_compare_exchange_unsigned(unsigned *addr,
                                                    unsigned compare,
                                                    unsigned val);
__NOOVL__ unsigned atomic_add_unsigned(unsigned *addr, unsigned val);
__NOOVL__ unsigned atomic_sub_unsigned(unsigned *addr, unsigned val);

__NOOVL__ int atomic_exchange_int(int *addr, int val);
__NOOVL__ int atomic_compare_exchange_int(int *addr, int compare, int val);
__NOOVL__ int atomic_add_int(int *addr, int val);
__NOOVL__ int atomic_sub_int(int *addr, int val);

__NOOVL__ float atomic_exchange_float(float *addr, float val);
__NOOVL__ float atomic_add_float(float *addr, float val);
__NOOVL__ float atomic_sub_float(float *addr, float val);

__NOOVL__ unsigned long long atomic_exchange_uint64(unsigned long long *addr,
                                                    unsigned long long val);
__NOOVL__ unsigned long long
atomic_compare_exchange_uint64(unsigned long long *addr,
                               unsigned long long compare,
                               unsigned long long val);
__NOOVL__ unsigned long long atomic_add_uint64(unsigned long long *addr,
                                               unsigned long long val);

__NOOVL__ unsigned atomic_and_unsigned(unsigned *addr, unsigned val);
__NOOVL__ unsigned atomic_or_unsigned(unsigned *addr, unsigned val);
__NOOVL__ unsigned atomic_xor_unsigned(unsigned *addr, unsigned val);
__NOOVL__ unsigned atomic_max_unsigned(unsigned *addr, unsigned val);
__NOOVL__ unsigned atomic_min_unsigned(unsigned *addr, unsigned val);

__NOOVL__ int atomic_and_int(int *addr, int val);
__NOOVL__ int atomic_or_int(int *addr, int val);
__NOOVL__ int atomic_xor_int(int *addr, int val);
__NOOVL__ int atomic_max_int(int *addr, int val);
__NOOVL__ int atomic_min_int(int *addr, int val);

__NOOVL__ unsigned long long atomic_and_uint64(unsigned long long *addr,
                                               unsigned long long val);
__NOOVL__ unsigned long long atomic_or_uint64(unsigned long long *addr,
                                              unsigned long long val);
__NOOVL__ unsigned long long atomic_xor_uint64(unsigned long long *addr,
                                               unsigned long long val);
__NOOVL__ unsigned long long atomic_max_uint64(unsigned long long *addr,
                                               unsigned long long val);
__NOOVL__ unsigned long long atomic_min_uint64(unsigned long long *addr,
                                               unsigned long long val);

__NOOVL__ unsigned atomic_inc_unsigned(unsigned *addr);
__NOOVL__ unsigned atomic_dec_unsigned(unsigned *addr);

__NOOVL__ int atomic_inc_int(int *addr);
__NOOVL__ int atomic_dec_int(int *addr);
#ifdef __cplusplus
}
#endif

#endif
