//===----- memoryheap.cu - NVPTX OpenMP debug utilities -------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of memory heap
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"
#include <stdio.h>

#ifdef __AMDGCN__

#define NUM_PAGES_PER_THREAD 16
#define SIZE_OF_PAGE 64
#define NUM_THREADS_PER_CU 64
#define NUM_CUS_PER_GPU 64
#define NUM_PAGES NUM_PAGES_PER_THREAD *NUM_THREADS_PER_CU *NUM_CUS_PER_GPU
#define SIZE_MALLOC NUM_PAGES *SIZE_OF_PAGE
#define SIZE_OF_HEAP SIZE_MALLOC

#ifndef nullptr
//#define nullptr((void *)0)
#define nullptr 0
#endif

#define hipBlockDim_x blockDim.x
#define hipBlockDim_y blockDim.y
#define hipBlockDim_z blockDim.z

#define hipBlockIdx_x blockIdx.x
#define hipBlockIdx_y blockIdx.y
#define hipBlockIdx_z blockIdx.z

#define hipGridDim_x gridDim.x
#define hipGridDim_y gridDim.y
#define hipGridDim_z gridDim.z

#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z

__device__ char gpuHeap[SIZE_OF_HEAP];
__device__ uint32_t gpuFlags[NUM_PAGES];

__device__ void *__malloc(size_t size) {
  char *heap = (char *)gpuHeap;
  if (size > SIZE_OF_HEAP) {
    return (void *)nullptr;
  }
  uint32_t totalThreads = hipBlockDim_x * hipGridDim_x * hipBlockDim_y *
                          hipGridDim_y * hipBlockDim_z * hipGridDim_z;
  uint32_t currentWorkItem = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

  uint32_t numHeapsPerWorkItem = NUM_PAGES / totalThreads;
  uint32_t heapSizePerWorkItem = SIZE_OF_HEAP / totalThreads;

  uint32_t stride = size / SIZE_OF_PAGE;
  uint32_t start = numHeapsPerWorkItem * currentWorkItem;

  uint32_t k = 0;

  while (gpuFlags[k] > 0) {
    k++;
  }

  for (uint32_t i = 0; i < stride - 1; i++) {
    gpuFlags[i + start + k] = 1;
  }

  gpuFlags[start + stride - 1 + k] = 2;

  void *ptr =
      (void *)(heap + heapSizePerWorkItem * currentWorkItem + k * SIZE_OF_PAGE);

  return ptr;
}

__device__ void __free(void *ptr) {
  if (ptr == nullptr) {
    return;
  }

  uint32_t offsetByte = (uint64_t)ptr - (uint64_t)gpuHeap;
  uint32_t offsetPage = offsetByte / SIZE_OF_PAGE;

  while (gpuFlags[offsetPage] != 0) {
    if (gpuFlags[offsetPage] == 2) {
      gpuFlags[offsetPage] = 0;
      offsetPage++;
      break;
    } else {
      gpuFlags[offsetPage] = 0;
      offsetPage++;
    }
  }

  return;
}

#endif
