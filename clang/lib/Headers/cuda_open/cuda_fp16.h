//===--------- cuda_open/cuda_fp16.h  -------------------------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef __CUDA_OPEN_FP16_HPP__
#define __CUDA_OPEN_FP16_HPP__
#ifndef __CUDA__
#error "This file is for CUDA compilation only."
#endif
#define __DEVICE__ __device__ __inline__ __attribute__((always_inline))
#define __HOST_DEVICE__                                                        \
  __host__ __device__ __inline__ __attribute__((always_inline))

#define half2half2 __half2half2
#define h2div __h2div

typedef _Float16 _Float16_2 __attribute__((ext_vector_type(2)));

struct __half_raw {
  union {
    static_assert(sizeof(_Float16) == sizeof(unsigned short), "");

    _Float16 data;
    unsigned short x;
  };
};

struct __half2_raw {
  union {
    static_assert(sizeof(_Float16_2) == sizeof(unsigned short[2]), "");

    _Float16_2 data;
    struct {
      unsigned short x;
      unsigned short y;
    };
  };
};

#if defined(__cplusplus)
#include "cuda_open/vector_types.h"

// BEGIN STRUCT __HALF
struct __half {
protected:
  union {
    static_assert(sizeof(_Float16) == sizeof(unsigned short), "");

    _Float16 data;
    unsigned short __x;
  };

public:
  // CREATORS
  __HOST_DEVICE__ __half() = default;
  __HOST_DEVICE__ __half(const __half_raw &x) : data{x.data} {}
  __HOST_DEVICE__ __half(decltype(data) x) : data{x} {}
  __HOST_DEVICE__ __half(const __half &) = default;
  __HOST_DEVICE__ __half(__half &&) = default;
  __HOST_DEVICE__ ~__half() = default;

  // MANIPULATORS
  __HOST_DEVICE__ __half &operator=(const __half &) = default;
  __HOST_DEVICE__ __half &operator=(__half &&) = default;
  __HOST_DEVICE__ __half &operator=(const __half_raw &x) {
    data = x.data;
    return *this;
  }
  __HOST_DEVICE__ volatile __half &operator=(const __half_raw &x) volatile {
    data = x.data;
    return *this;
  }
  __HOST_DEVICE__ volatile __half &
  operator=(const volatile __half_raw &x) volatile {
    data = x.data;
    return *this;
  }
  __HOST_DEVICE__ __half &operator=(__half_raw &&x) {
    data = x.data;
    return *this;
  }
  __HOST_DEVICE__ volatile __half &operator=(__half_raw &&x) volatile {
    data = x.data;
    return *this;
  }

  __HOST_DEVICE__ volatile __half &operator=(volatile __half_raw &&x) volatile {
    data = x.data;
    return *this;
  }

  __HOST_DEVICE__ operator int() { return static_cast<int>(data); }

  __DEVICE__ __half &operator+=(const __half &x) {
    data += x.data;
    return *this;
  }
  __DEVICE__ __half &operator-=(const __half &x) {
    data -= x.data;
    return *this;
  }
  __DEVICE__ __half &operator*=(const __half &x) {
    data *= x.data;
    return *this;
  }
  __DEVICE__ __half &operator/=(const __half &x) {
    data /= x.data;
    return *this;
  }
  __DEVICE__ __half &operator++() {
    ++data;
    return *this;
  }
  __DEVICE__
  __half operator++(int) {
    __half tmp{*this};
    ++*this;
    return tmp;
  }
  __DEVICE__ __half &operator--() {
    --data;
    return *this;
  }
  __DEVICE__ __half operator--(int) {
    __half tmp{*this};
    --*this;
    return tmp;
  }

  __HOST_DEVICE__ operator __half_raw() const { return __half_raw{data}; }
  __HOST_DEVICE__ operator volatile __half_raw() const volatile {
    return __half_raw{data};
  }

  __DEVICE__ __half operator+() const { return *this; }
  __DEVICE__ __half operator-() const {
    __half tmp{*this};
    tmp.data = -tmp.data;
    return tmp;
  }
};

struct __half2 {
protected:
  union {
    static_assert(sizeof(_Float16_2) == sizeof(unsigned short[2]), "");

    _Float16_2 data;
    struct {
      unsigned short x;
      unsigned short y;
    };
  };

public:
  // CREATORS
  __HOST_DEVICE__ __half2() = default;
  __HOST_DEVICE__ __half2(const __half2_raw &x) : data{x.data} {}
  __HOST_DEVICE__ __half2(decltype(data) x) : data{x} {}
  __HOST_DEVICE__ __half2(const __half &x, const __half &y)
      : data{static_cast<__half_raw>(x).data, static_cast<__half_raw>(y).data} {
  }
  __HOST_DEVICE__ __half2(const __half2 &) = default;
  __HOST_DEVICE__ __half2(__half2 &&) = default;
  __HOST_DEVICE__ ~__half2() = default;

  // MANIPULATORS
  __HOST_DEVICE__ __half2 &operator=(const __half2 &) = default;
  __HOST_DEVICE__ __half2 &operator=(__half2 &&) = default;
  __HOST_DEVICE__ __half2 &operator=(const __half2_raw &x) {
    data = x.data;
    return *this;
  }

  __DEVICE__ __half2 &operator+=(const __half2 &x) {
    data += x.data;
    return *this;
  }
  __DEVICE__ __half2 &operator-=(const __half2 &x) {
    data -= x.data;
    return *this;
  }
  __DEVICE__ __half2 &operator*=(const __half2 &x) {
    data *= x.data;
    return *this;
  }
  __DEVICE__ __half2 &operator/=(const __half2 &x) {
    data /= x.data;
    return *this;
  }
  __DEVICE__ __half2 &operator++() { return *this += _Float16_2{1, 1}; }
  __DEVICE__ __half2 operator++(int) {
    __half2 tmp{*this};
    ++*this;
    return tmp;
  }
  __DEVICE__ __half2 &operator--() { return *this -= _Float16_2{1, 1}; }
  __DEVICE__ __half2 operator--(int) {
    __half2 tmp{*this};
    --*this;
    return tmp;
  }

  // ACCESSORS
  __HOST_DEVICE__ operator decltype(data)() const { return data; }
  __HOST_DEVICE__ operator __half2_raw() const { return __half2_raw{data}; }

  // ACCESSORS - DEVICE ONLY
  __DEVICE__ __half2 operator+() const { return *this; }
  __DEVICE__ __half2 operator-() const {
    __half2 tmp{*this};
    tmp.data = -tmp.data;
    return tmp;
  }
};
// END STRUCT __HALF2
#endif // defined(__cplusplus)

__HOST_DEVICE__ __half __float2half(float a);
__HOST_DEVICE__ float __half2float(__half a);

__DEVICE__ __half __hadd(__half x, __half y) {
  return __half_raw{static_cast<__half_raw>(x).data +
                    static_cast<__half_raw>(y).data};
}

#ifndef __USE_OPEN_HEADERS__
#error "This file requires -D__USE_OPEN_HEADERS__"
#endif

#endif
