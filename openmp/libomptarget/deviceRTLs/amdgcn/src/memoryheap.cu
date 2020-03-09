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

#include "common/omptarget.h"
#include "amdgcn_access_dimensions.h"
#include <stdio.h>

#ifdef __AMDGCN__

#define NUM_PAGES_PER_THREAD 16
#define SIZE_OF_PAGE 64
#define NUM_THREADS_PER_CU  64  // should be 1024 ???
#define NUM_CUS_PER_GPU 64
#define NUM_PAGES NUM_PAGES_PER_THREAD *NUM_THREADS_PER_CU *NUM_CUS_PER_GPU
#define SIZE_MALLOC NUM_PAGES *SIZE_OF_PAGE
#define SIZE_OF_HEAP SIZE_MALLOC

#ifndef nullptr
#define nullptr 0
#endif

#define hipBlockDim_x __amdgcn_block_dim_x()
#define hipBlockDim_y __amdgcn_block_dim_y()
#define hipBlockDim_z __amdgcn_block_dim_z()

#define hipBlockIdx_x __builtin_amdgcn_workgroup_id_x()

#define hipGridDim_x __amdgcn_grid_dim_x()
#define hipGridDim_y __amdgcn_grid_dim_y()
#define hipGridDim_z __amdgcn_grid_dim_z()

#define hipThreadIdx_x __builtin_amdgcn_workitem_id_x()

DEVICE char gpuHeap[SIZE_OF_HEAP];
DEVICE uint32_t gpuFlags[NUM_PAGES];

DEVICE void *__malloc(size_t size) {
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

DEVICE void __free(void *ptr) {
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
