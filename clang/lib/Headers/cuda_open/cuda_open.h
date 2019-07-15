//===------------------------ cuda_open.h  -------------------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//===----------------------------------------------------------------------===//

// Based on the CUDA RUNTIME API documentation.  v9.1.85
// http://docs.nvidia.com/cuda/pdf/CUDA_Runtime_API.pdf

#ifndef __CLANG__CUDA_OPEN_H__
#define __CLANG__CUDA_OPEN_H__
#ifndef __CUDA__
#error "This file is for CUDA compilation only."
#endif

// This is the first file included by __clang_cuda_runtime_wrapper_open.h.
// Clang auto includes __clang_cuda_runtime_wrapper_open.h when the clang
// driver determines there are no nvidia targets.
// If there are nvidia targets then the driver checks for the Nvidia cuda SDK
// and if successful, it auto includes  __clang_cuda_runtime_wrapper.h
// which includes protected headers from the Nvidia cuda sdk such as cuda.h.
// You will notice that __clang_cuda_runtime_wrapper_open.h is very similar to
// __clang_cuda_runtime_wrapper.h. The major difference is that it explicitly
// includes files in this directory (cuda_open) with similar names to those
// expected from the nvidia cuda sdk with the exception of cuda.h. Instead of
// of cuda.h,  __clang_cuda_runtime_wraper_open.h includes
// <cuda_open/cuda_open.h. Currently, most of the files in this cuda_open
// directory are empty stubs. A redistribution of the content and some
// simplification of the file structure in this cuda_open directory will
// occur after sufficient review by interested parties.

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __managed__ __attribute__((managed))
#define __host__ __attribute__((host))
#define __constant__ __attribute__((constant))
#define __noinline__ __attribute__((noinline))
#define __forceinline__ __attribute__((always_inline))
#define __align__(n) __attribute__((aligned(n)))

#include "cuda_open/vector_types.h"

struct dim3 {
  unsigned int x, y, z;
#if defined(__cplusplus)
  __host__ __device__ dim3(unsigned int X = 1, unsigned int Y = 1,
                           unsigned int Z = 1)
      : x(X), y(Y), z(Z) {}
  __host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  __host__ __device__ operator uint3(void) {
    uint3 t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
  }
#endif /
};
typedef struct dim3 dim3;

// 4.31. Data types used by CUDA Runtime
#define CUDA_EGL_MAX_PLANES 3
#define CUDA_IPC_HANDLE_SIZE 64
#define cudaArrayCubemap 0x04
#define cudaArrayDefault 0x00
#define cudaArrayLayered 0x01
#define cudaArraySurfaceLoadStore 0x02
#define cudaArrayTextureGather 0x08
#define cudaCooperativeLaunchMultiDeviceNoPostSync 0x02
#define cudaCooperativeLaunchMultiDeviceNoPreSync 0x01
#define cudaCpuDeviceId ((int)-1)
#define cudaDeviceBlockingSync 0x04
#define cudaDeviceLmemResizeToMax 0x10
#define cudaDeviceMapHost 0x08
#define cudaDeviceMask 0x1f
#define cudaDevicePropDontCare
#define cudaDeviceScheduleAuto 0x00
#define cudaDeviceScheduleBlockingSync 0x04
#define cudaDeviceScheduleMask 0x07
#define cudaDeviceScheduleSpin 0x01
#define cudaDeviceScheduleYield 0x02
#define cudaEventBlockingSync 0x01
#define cudaEventDefault 0x00
#define cudaEventDisableTiming 0x02
#define cudaEventInterprocess 0x04
#define cudaHostAllocDefault 0x00
#define cudaHostAllocMapped 0x02
#define cudaHostAllocPortable 0x01
#define cudaHostAllocWriteCombined 0x04
#define cudaHostRegisterDefault 0x00
#define cudaHostRegisterIoMemory 0x04
#define cudaHostRegisterMapped 0x02
#define cudaHostRegisterPortable 0x01
#define cudaInvalidDeviceId ((int)-2)
#define cudaIpcMemLazyEnablePeerAccess 0x01
#define cudaMemAttachGlobal 0x01
#define cudaMemAttachHost 0x02
#define cudaMemAttachSingle 0x04
#define cudaOccupancyDefault 0x00
#define cudaOccupancyDisableCachingOverride 0x01
#define cudaPeerAccessDefault 0x00
#define cudaStreamDefault 0x00
#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamNonBlocking 0x01
#define cudaStreamPerThread ((cudaStream_t)0x2)

#if defined(__cplusplus)
extern "C" {
#endif

struct cudaChannelFormatDesc;
struct cudaDeviceProp;
struct cudaEglFrame;
struct cudaExtent;
struct cudaFuncAttributes;
struct cudaIpcEventHandle_t;
struct cudaIpcMemHandle_t;
struct cudaLaunchParams;
struct cudaMemcpy3DParms;
struct cudaMemcpy3DPeerParms;
struct cudaPitchedPtr;
struct cudaPointerAttributes;
struct cudaPos;
struct cudaResourceDesc;
struct cudaResourceViewDesc;
struct cudaTextureDesc;
struct surfaceReference;
struct textureReference;

enum cudaCGScope {
  cudaCGScopeInvalid = 0,
  cudaCGScopeGrid = 1,
  cudaCGScopeMultiGrid = 2
};

enum cudaComputeMode {
  cudaComputeModeDefault = 0,
  cudaComputeModeExclusive = 1,
  cudaComputeModeProhibited = 2,
  cudaComputeModeExcusiveProcess = 3
};

enum cudaDeviceP2PAttr {
  cudaDevP2PAttrPerformanceRank = 1,
  cudaDevP2PAttrAccessSupported = 2,
  cudaDevP2PAttrNativeAtomicSupported = 3
};

enum cudaEglResourceLocationFlags {
  cudaEglResourceLocationSysmem = 0x00,
  cudaEglResourceLocationVidmem = 0x01
};

enum cudaFuncAttribute {
  cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
  cudaFuncAttributePreferredSharedMemoryCarveout = 9,
  cudaFuncAttributeMax
};

enum cudaGraphicsCubeFace {
  cudaGraphicsCubeFacePositiveX = 0x00,
  cudaGraphicsCubeFaceNegativeX = 0x01,
  cudaGraphicsCubeFacePositiveY = 0x02,
  cudaGraphicsCubeFaceNegativeY = 0x03,
  cudaGraphicsCubeFacePositiveZ = 0x04,
  cudaGraphicsCubeFaceNegativeZ = 0x05
};

enum cudaGraphicsMapFlags {
  cudaGraphicsMapFlagsNone = 0,
  cudaGraphicsMapFlagsReadOnly = 1,
  cudaGraphicsMapFlagsWriteDiscard = 2
};

enum cudaGraphicsRegisterFlags {
  cudaGraphicsRegisterFlagsNone = 0,
  cudaGraphicsRegisterFlagsReadOnly = 1,
  cudaGraphicsRegisterFlagsWriteDiscard = 2,
  cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
  cudaGraphicsRegisterFlagsTextureGather = 8
};

enum cudaLimit {
  cudaLimitStackSize = 0x00,
  cudaLimitPrintfFifoSize = 0x01,
  cudaLimitMallocHeapSize = 0x02,
  cudaLimitDevRuntimeSyncDepth = 0x03,
  cudaLimitDevRuntimePendingLaunchCount = 0x04
};

enum cudaMemoryAdvise {
  cudaMemAdviseSetReadMostly = 1,
  cudaMemAdviseUnsetReadMostly = 2,
  cudaMemAdviseSetPreferredLocation = 3,
  cudaMemAdviseUnsetPreferredLocation = 4,
  cudaMemAdviseSetAccessedBy = 5,
  cudaMemAdviseUnsetAccessedBy = 6
};

enum cudaMemRangeAttribute {
  cudaMemRangeAttributeReadMostly = 1,
  cudaMemRangeAttributePreferredLocation = 2,
  cudaMemRangeAttributeAccessedBy = 3,
  cudaMemRangeAttributeLastPrefetchLocation = 4
};

enum cudaSharedCarveout {
  cudaSharedmemCarveoutDefault = -1,
  cudaSharedmemCarveoutMaxShared = 100,
  cudaSharedmemCarveoutMaxL1 = 0
};

enum cudaSharedMemConfig {
  cudaSharedMemBankSizeDefault = 0,
  cudaSharedMemBankSizeFourByte = 1,
  cudaSharedMemBankSizeEightByte = 2
};

enum cudaSurfaceBoundaryMode {
  cudaBoundaryModeZero = 0,
  cudaBoundaryModeClamp = 1,
  cudaBoundaryModeTrap = 2
};

enum cudaSurfaceFormatMode { cudaFormatModeForced = 0, cudaFormatModeAuto = 1 };

// Chapter 5 DATA STRUCTURES
enum cudaError {
  cudaSuccess = 0,
  cudaErrorMissingConfiguration = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorLaunchFailure = 4,
  cudaErrorPriorLaunchFailure = 5, // Deprecated
  cudaErrorLaunchTimeout = 6,
  cudaErrorLaunchOutOfResources = 7,
  cudaErrorInvalidDeviceFunction = 8,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidDevice = 10,
  cudaErrorInvalidValue = 11,
  cudaErrorInvalidPitchValue = 12,
  cudaErrorInvalidSymbol = 13,
  cudaErrorMapBufferObjectFailed = 14,
  cudaErrorUnmapBufferObjectFailed = 15,
  cudaErrorInvalidHostPointer = 16,
  cudaErrorINvalidDevicePointer = 17,
  cudaErrorInvalidTexture = 18,
  cudaErrorInvalidTextureBinding = 19,
  cudaErrorInvalidChannelDescriptor = 20,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorAddressOfConstant = 22,
  cudaErrorTextureFetchFailed = 23,
  cudaErrorTextureNotBound = 24,
  cudaErrorSynchronizationError = 25,
  cudaErrorInvalidFilterSetting = 26,
  cudaErrorInvalidNormSetting = 27,
  cudaErrorMixedDeviceExecution = 28,
  cudaErrorCudartUnloading = 29,
  cudaErrorUnknown = 30,
  cudaErrorNotYetImplemented = 31,
  cudaErrorMemoryValueTooLarge = 32,
  cudaErrorInvalidResourceHandle = 33,
  cudaErrorNotReady = 34,
  cudaErrorInsufficientDriver = 35,
  cudaErrorSetOnActiveProcess = 36,
  cudaErrorInvalidSurface = 37,
  cudaErrorNoDevice = 38,
  cudaErrorECCUncorrectable = 39,
  cudaErrorSharedObjectSymbolNotFound = 40,
  cudaErrorSharedObjectInitFailed = 41,
  cudaErrorUnsupportedLimit = 42,
  cudaErrorDuplicateVariableName = 43,
  cudaErrorDuplicateTextureName = 44,
  cudaErrorDuplicateSurfaceName = 45,
  cudaErrorDevicesUnavailable = 46,
  cudaErrorInvalidKernelImage = 47,
  cudaErrorNoKernelImageForDevice = 48,
  cudaErrorIncompatibleDriverContext = 49,
  cudaErrorPeerAccessAlreadyEnabled = 50,
  cudaErrorPeerAccessNotEnabled = 51,
  cudaErrorDeviceAlreadyInUse = 54,
  cudaErrorProfilerDisabled = 55,
  cudaErrorProfilerNotInitialized = 56,
  cudaErrorProfilerAlreadyStarted = 57,
  cudaErrorProfilerAlreadyStopped = 58,
  cudaErrorAssert = 59,
  cudaErrorTooManyPeers = 60,
  cudaErrorHostMemoryAlreadyRegistered = 61,
  cudaErrorHostMemoryNotRegistered = 62,
  cudaErrorOperatingSystem = 63,
  cudaErrorPeerAccessUnsupported = 64,
  cudaErrorLaunchMaxDepthExceeded = 65,
  cudaErrorLaunchFileScopedTex = 66,
  cudaErrorLaunchFileScopedSurf = 67,
  cudaErrorSyncDepthExceeded = 68,
  cudaErrorLaunchPendingCountExceeded = 69,
  cudaErrorNotPermitted = 70,
  cudaErrorNotSupported = 71,
  cudaErrorHardwareStackError = 72,
  cudaErrorIllegalInstruction = 73,
  cudaErrorMisalignedAddress = 74,
  cudaErrorInvalidAddressSpace = 75,
  cudaErrorInvalidPc = 76,
  cudaErrorIllegalAddress = 77,
  cudaErrorInvalidPtx = 78,
  cudaErrorInvalidGraphicsContext = 79,
  cudaErrorNvlinkUncorrectable = 80,
  cudaErrorJitCompilerNotFound = 81,
  cudaErrorCooperativeLaunchTooLarge = 82,
  cudaErrorStartupFailure = 0x7f,
  cudaErrorApiFailureBase = 10000
};

enum cudaOutputMode { cudaKeyValuePair = 0x00, cudaCSV = 0x01 };

struct cudaArray;
struct cudaGraphicsResource;
struct cudaMipmappedArray;

typedef cudaArray *cudaArray_const_t; // FIXME: Check if this is correct (does
                                      // not have const in doc)
typedef cudaArray *cudaArray_t;
typedef struct CUeglStreamConnection_st *cudaEglStreamConnection;
typedef cudaError cudaError_t;
typedef struct CUevent_st *cudaEvent_t;
typedef cudaGraphicsResource *cudaGraphicsResource_t;
typedef cudaMipmappedArray *cudaMipmappedArray_t;
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t;
typedef cudaOutputMode cudaOutputMode_t;
typedef struct CUstream_st *cudaStream_t;
typedef unsigned long long cudaSurfaceObject_t;
typedef unsigned long long cudaTextureObject_t;
typedef struct CUuuid_stcudaUUID_t;

enum cudaChannelFormatKind {
  cudaChannelFormatKindSigned = 0,
  cudaChannelFormatKindUnsigned = 1,
  cudaChannelFormatKindFloat = 2,
  cudaChannelFormatKindNone = 3
};

// 5.1. __cudaOccupancyB2DHelper

// 5.2. cudaChannelFormatDesc Struct Reference
struct cudaChannelFormatDesc { // FIXME: Verify this
  enum cudaChannelFormatKind f;
  int w;
  int x;
  int y;
  int z;
};

// 5.3 cudaDeviceProp Struct Reference

enum cudaEglColorFormat {
  cudaEglColorFormatYUV420Planar = 0,
  cudaEglColorFormatYUV420SemiPlanar = 1,
  cudaEglColorFormatYUV422Planar = 2,
  cudaEglColorFormatYUV422SemiPlanar = 3,
  cudaEglColorFormatRGB = 4,
  cudaEglColorFormatBGR = 5,
  cudaEglColorFormatARGB = 6,
  cudaEglColorFormatRGBA = 7,
  cudaEglColorFormatL = 8,
  cudaEglColorFormatR = 9,
  cudaEglColorFormatYUV444Planar = 10,
  cudaEglColorFormatYUV444SemiPlanar = 11,
  cudaEglColorFormatYUYV422 = 12,
  cudaEglColorFormatUYVY422 = 13,
  cudaEglColorFormatABGR = 14,
  cudaEglColorFormatBGRA = 15,
  cudaEglColorFormatA = 16,
  cudaEglColorFormatRG = 17,
  cudaEglColorFormatAYUV = 18,
  cudaEglColorFormatYVU444SemiPlanar = 19,
  cudaEglColorFormatYVU422SemiPlanar = 20,
  cudaEglColorFormatYVU420SemiPlanar = 21,
  cudaEglColorFormatY10V10U10_444SemiPlanar = 22,
  cudaEglColorFormatY10V10U10_420SemiPlanar = 23,
  cudaEglColorFormatY12V12U12_444SemiPlanar = 24,
  cudaEglColorFormatY12V12U12_420SemiPlanar = 25,
  cudaEglColorFormatVYUY_ER = 26,
  cudaEglColorFormatUYVY_ER = 27,
  cudaEglColorFormatYUYV_ER = 28,
  cudaEglColorFormatYVYU_ER = 29,
  cudaEglColorFormatYUV_ER = 30,
  cudaEglColorFormatYUVA_ER = 31,
  cudaEglColorFormatAYUV_ER = 32,
  cudaEglColorFormatYUV444Planar_ER = 33,
  cudaEglColorFormatYUV422Planar_ER = 34,
  cudaEglColorFormatYUV420Planar_ER = 35,
  cudaEglColorFormatYUV444SemiPlanar_ER = 36,
  cudaEglColorFormatYUV422SemiPlanar_ER = 37,
  cudaEglColorFormatYUV420SemiPlanar_ER = 38,
  cudaEglColorFormatYVU444Planar_ER = 39,
  cudaEglColorFormatYVU422Planar_ER = 40,
  cudaEglColorFormatYVU420Planar_ER = 41,
  cudaEglColorFormatYVU444SemiPlanar_ER = 42,
  cudaEglColorFormatYVU422SemiPlanar_ER = 43,
  cudaEglColorFormatYVU420SemiPlanar_ER = 44,
  cudaEglColorFormatBayerRGGB = 45,
  cudaEglColorFormatBayerBGGR = 46,
  cudaEglColorFormatBayerGRBG = 47,
  cudaEglColorFormatBayerGBRG = 48,
  cudaEglColorFormatBayer10RGGB = 49,
  cudaEglColorFormatBayer10BGGR = 50,
  cudaEglColorFormatBayer10GRBG = 51,
  cudaEglColorFormatBayer10GBRG = 52,
  cudaEglColorFormatBayer12RGGB = 53,
  cudaEglColorFormatBayer12BGGR = 54,
  cudaEglColorFormatBayer12GRBG = 55,
  cudaEglColorFormatBayer12GBRG = 56,
  cudaEglColorFormatBayer14RGGB = 57,
  cudaEglColorFormatBayer14BGGR = 58,
  cudaEglColorFormatBayer14GRBG = 59,
  cudaEglColorFormatBayer14GBRG = 60,
  cudaEglColorFormatBayer20RGGB = 61,
  cudaEglColorFormatBayer20BGGR = 62,
  cudaEglColorFormatBayer20GRBG = 63,
  cudaEglColorFormatBayer20GBRG = 64,
  cudaEglColorFormatYVU444Planar = 65,
  cudaEglColorFormatYVU422Planar = 66,
  cudaEglColorFormatYVU420Planar = 67
};

enum cudaEglFrameType { cudaEglFrameTypeArray = 0, cudaEglFrameTypePitch = 1 };

// 5.13. cudaPitchedPtr Struct Reference
struct cudaPitchedPtr {
  size_t pitch;
  void *ptr;
  size_t xsize;
  size_t ysize;
};
/////

// 5.4. cudaEglFrame Struct Reference

// 5.5. cudaEglPlaneDesc Struct Reference
typedef struct cudaEglPlaneDesc_st {
  unsigned int width;
  unsigned int height;
  unsigned int depth;
  unsigned int pitch;
  unsigned int numChannels;
  struct cudaChannelFormatDesc channelDesc;
  unsigned int reserved[4];
} cudaEglPlaneDesc;

struct cudaEglFrame {
  cudaEglColorFormat eglColorFormat;
  cudaEglFrameType frameType;
  cudaArray_t pArray;
  unsigned int planeCount;
  cudaEglPlaneDesc planeDesc;
  struct cudaPitchedPtr pPitch;
};

// 5.6. cudaExtent Struct Reference
struct cudaExtent {
  size_t depth;
  size_t height;
  size_t width;
};

// 5.7. cudaFuncAttributes Struct Reference
struct cudaFuncAttributes {
  int binaryVersion;
  int cacheModeCA;
  size_t constSizeBytes;
  size_t localSizeBytes;
  int maxDynamicSharedSizeBytes;
  int maxThreadsPerBlock;
  int numRegs;
  int preferredShmemCarveout;
  int ptxVersion;
  size_t sharedSizeBytes;
};

// 5.8. cudaIpcEventHandle_t Struct Reference
// CUDA IPC event handle

// 5.9. cudaIpcMemHandle_t Struct Reference
// CUDA IPC memory handle

// 5.10. cudaLaunchParams Struct Reference
struct cudaLaunchParams {
  void *func;
  dim3 gridDim;
  dim3 blockDim;
  void **args;
  size_t sharedMem;
  cudaStream_t stream;
};

// 5.15. cudaPos Struct Reference
struct cudaPos {
  size_t x;
  size_t y;
  size_t z;
};

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};

// 5.11. cudaMemcpy3DParms Struct Reference
struct cudaMemcpy3DParms {
  cudaArray_t dstArray;
  struct cudaPos dstPos;
  struct cudaPitchedPtr dstPtr;
  struct cudaExtent extent;
  cudaMemcpyKind kind;
  cudaArray_t srcArray;
  struct cudaPos srcPos;
  struct cudaPitchedPtr srcPtr;
};

// 5.12. cudaMemcpy3DPeerParms Struct Reference
struct cudaMemcpy3DPeerParms {
  cudaArray_t dstArray;
  int dstDevice;
  struct cudaPos dstPos;
  struct cudaPitchedPtr dstPtr;
  struct cudaExtent extent;
  cudaArray_t srcArray;
  int srcDevice;
  struct cudaPos srcPos;
  struct cudaPitchedPtr srcPtr;
};

// 5.14. cudaPointerAttributes Struct Reference
enum cudaMemoryType { cudaMemoryTypeHost = 1, cudaMemoryTypeDevice = 2 };

struct cudaPointerAttributes {
  int device;
  void *devicePointer;
  void *hostPointer;
  int isManaged;
  enum cudaMemoryType memoryType;
};

// 5.16. cudaResourceDesc Struct Reference
enum cudaResourceType {
  cudaResourceTypeArray = 0x00,
  cudaResourceTypeMipmappedArray = 0x01,
  cudaResourceTypeLinear = 0x02,
  cudaResourceTypePitch2D = 0x03
};

struct cudaResourceDesc {
  cudaArray_t array;
  struct cudaChannelFormatDesc desc;
  void *devPtr;
  size_t height;
  cudaMipmappedArray_t mipmap;
  size_t pitchInBytes;
  cudaResourceType resType;
  size_t sizeInBytes;
  size_t width;
};

// 5.17. cudaResourceViewDesc Struct Reference
enum cudaResourceViewFormat {
  cudaResViewFormatNone = 0x00,
  cudaResViewFormatUnsignedChar1 = 0x01,
  cudaResViewFormatUnsignedChar2 = 0x02,
  cudaResViewFormatUnsignedChar4 = 0x03,
  cudaResViewFormatSignedChar1 = 0x04,
  cudaResViewFormatSignedChar2 = 0x05,
  cudaResViewFormatSignedChar4 = 0x06,
  cudaResViewFormatUnsignedShort1 = 0x07,
  cudaResViewFormatUnsignedShort2 = 0x08,
  cudaResViewFormatUnsignedShort4 = 0x09,
  cudaResViewFormatSignedShort1 = 0x0a,
  cudaResViewFormatSignedShort2 = 0x0b,
  cudaResViewFormatSignedShort4 = 0x0c,
  cudaResViewFormatUnsignedInt1 = 0x0d,
  cudaResViewFormatUnsignedInt2 = 0x0e,
  cudaResViewFormatUnsignedInt4 = 0x0f,
  cudaResViewFormatSignedInt1 = 0x10,
  cudaResViewFormatSignedInt2 = 0x11,
  cudaResViewFormatSignedInt4 = 0x12,
  cudaResViewFormatHalf1 = 0x13,
  cudaResViewFormatHalf2 = 0x14,
  cudaResViewFormatHalf4 = 0x15,
  cudaResViewFormatFloat1 = 0x16,
  cudaResViewFormatFloat2 = 0x17,
  cudaResViewFormatFloat4 = 0x18,
  cudaResViewFormatUnsignedBlockCompressed1 = 0x19,
  cudaResViewFormatUnsignedBlockCompressed2 = 0x1a,
  cudaResViewFormatUnsignedBlockCompressed3 = 0x1b,
  cudaResViewFormatUnsignedBlockCompressed4 = 0x1c,
  cudaResViewFormatSignedBlockCompressed4 = 0x1d,
  cudaResViewFormatUnsignedBlockCompressed5 = 0x1e,
  cudaResViewFormatSignedBlockCompressed5 = 0x1f,
  cudaResViewFormatUnsignedBlockCompressed6H = 0x20,
  cudaResViewFormatSignedBlockCompressed6H = 0x21,
  cudaResViewFormatUnsignedBlockCompressed7 = 0x22
};

struct cudaResourceViewDesc {
  size_t depth;
  unsigned int firstLayer;
  unsigned int firstMipmapLevel;
  enum cudaResourceViewFormat format;
  size_t height;
  unsigned int lastLayer;
  unsigned int lastMipmapLevel;
  size_t width;
};

// 5.18 cudaTextureDesc Struct Reference
enum cudaTextureAddressMode {
  cudaAddressModeWrap = 0,
  cudaAddressModeClamp = 1,
  cudaAddressModeMirror = 2,
  cudaAddressModeBorder = 3
};

enum cudaTextureFilterMode {
  cudaFilterModePoint = 0,
  cudaFilterModeLinear = 1
};

enum cudaTextureReadMode {
  cudaReadModeElementType = 0,
  cudaReadModeNormalizedFloat = 1
};

struct cudaTextureDesc {
  enum cudaTextureAddressMode addressMode;
  float borderColor;
  enum cudaTextureFilterMode filterMode;
  unsigned int maxAnisotropy;
  float maxMipmapLevelClamp;
  float minMipmapLevelClamp;
  enum cudaTextureFilterMode mipmapFilterMode;
  float mipmapLevelBias;
  int normalizedCoords;
  enum cudaTextureReadMode readMode;
  int sRGB;
};

// 5.19. surfaceReference Struct Reference
struct surcafeReference {
  struct cudaChannelFormatDesc channelDesc;
};

// 5.20. textureReference Struct Reference
struct textureReference {
  enum cudaTextureAddressMode addressMode;
  struct cudaChannelFormatDesc channelDesc;
  enum cudaTextureFilterMode filterMode;
  unsigned int maxAnisotropy;
  float maxMipmapLevelClamp;
  float minMipmapLevelClamp;
  enum cudaTextureFilterMode mipmapFilterMode;
  float mipmapLevelBias;
  int normalized;
  int sRGB;
};

///////////////////////////////////////////////////////////////////////////////
// 4. Modules

// 4.1. Device Management
struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture1DMipmap;
  int maxTexture1DLinear;
  int maxTexture2D[2];
  int maxTexture2DMipmap[2];
  int maxTexture2DLinear[3];
  int maxTexture2DGather[2];
  int maxTexture3D[3];
  int maxTexture3DAlt[3];
  int maxTextureCubemap;
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  int maxTextureCubemapLayered[2];
  int maxSurface1D;
  int maxSurface2D[2];
  int maxSurface3D[3];
  int maxSurface1DLayered[2];
  int maxSurface2DLayered[3];
  int maxSurfaceCubemap;
  int maxSurfaceCubemapLayered[2];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int maxThreadsPerMultiProcessor;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemSupported;
  int isMultiGpuBoard;
  int multiGpuBoardGroupID;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
};

enum cudaDeviceAttr {
  cudaDevAttrMaxThreadsPerBlock = 1,
  cudaDevAttrMaxBlockDimX = 2,
  cudaDevAttrMaxBlockDimY = 3,
  cudaDevAttrMaxBlockDimZ = 4,
  cudaDevAttrMaxGridDimX = 5,
  cudaDevAttrMaxGridDimY = 6,
  cudaDevAttrMaxGridDimZ = 7,
  cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrTotalConstantMemory = 9,
  cudaDevAttrWarpSize = 10,
  cudaDevAttrMaxPitch = 11,
  cudaDevAttrMaxRegistersPerBlock = 12,
  cudaDevAttrClockRate = 13,
  cudaDevAttrTextureAlignment = 14,
  cudaDevAttrGpuOverlap = 15,
  cudaDevAttrMultiProcessorCount = 16,
  cudaDevAttrKernelExecTimeout = 17,
  cudaDevAttrIntegrated = 18,
  cudaDevAttrCanMapHostMemory = 19,
  cudaDevAttrComputeMode = 20,
  cudaDevAttrMaxTexture1DWidth = 21,
  cudaDevAttrMaxTexture2DWidth = 22,
  cudaDevAttrMaxTexture2DHeight = 23,
  cudaDevAttrMaxTexture3DWidth = 24,
  cudaDevAttrMaxTexture3DHeight = 25,
  cudaDevAttrMaxTexture3DDepth = 26,
  cudaDevAttrMaxTexture2DLayeredWidth = 27,
  cudaDevAttrMaxTexture2DLayeredHeight = 28,
  cudaDevAttrMaxTexture2DLayeredLayers = 29,
  cudaDevAttrSurfaceAlignment = 30,
  cudaDevAttrConcurrentKernels = 31,
  cudaDevAttrEccEnabled = 32,
  cudaDevAttrPciBusId = 33,
  cudaDevAttrPciDeviceId = 34,
  cudaDevAttrTccDriver = 35,
  cudaDevAttrMemoryClockRate = 36,
  cudaDevAttrGlobalMemoryBusWidth = 37,
  cudaDevAttrL2CacheSize = 38,
  cudaDevAttrMaxThreadsPerMultiProcessor = 39,
  cudaDevAttrAsyncEngineCount = 40,
  cudaDevAttrUnifiedAddressing = 41,
  cudaDevAttrMaxTexture1DLayeredWidth = 42,
  cudaDevAttrMaxTexture1DLayeredLayers = 43,
  cudaDevAttrMaxTexture2DGatherWidth = 45,
  cudaDevAttrMaxTexture2DGatherHeight = 46,
  cudaDevAttrMaxTexture3DWidthAlt = 47,
  cudaDevAttrMaxTexture3DHeightAlt = 48,
  cudaDevAttrMaxTexture3DDepthAlt = 49,
  cudaDevAttrPciDomainId = 50,
  cudaDevAttrTexturePitchAlignment = 51,
  cudaDevAttrMaxTextureCubemapWidth = 52,
  cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
  cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
  cudaDevAttrMaxSurface1DWidth = 55,
  cudaDevAttrMaxSurface2DWidth = 56,
  cudaDevAttrMaxSurface2DHeight = 57,
  cudaDevAttrMaxSurface3DWidth = 58,
  cudaDevAttrMaxSurface3DHeight = 59,
  cudaDevAttrMaxSurface3DDepth = 60,
  cudaDevAttrMaxSurface1DLayeredWidth = 61,
  cudaDevAttrMaxSurface1DLayeredLayers = 62,
  cudaDevAttrMaxSurface2DLayeredWidth = 63,
  cudaDevAttrMaxSurface2DLayeredHeight = 64,
  cudaDevAttrMaxSurface2DLayeredLayers = 65,
  cudaDevAttrMaxSurfaceCubemapWidth = 66,
  cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
  cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
  cudaDevAttrMaxTexture1DLinearWidth = 69,
  cudaDevAttrMaxTexture2DLinearWidth = 70,
  cudaDevAttrMaxTexture2DLinearHeight = 71,
  cudaDevAttrMaxTexture2DLinearPitch = 72,
  cudaDevAttrMaxTexture2DMipmappedWidth = 73,
  cudaDevAttrMaxTexture2DMipmappedHeight = 74,
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76,
  cudaDevAttrMaxTexture1DMipmappedWidth = 77,
  cudaDevAttrStreamPrioritiesSupported = 78,
  cudaDevAttrGlobalL1CacheSupported = 79,
  cudaDevAttrLocalL1CacheSupported = 80,
  cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
  cudaDevAttrMaxRegistersPerMultiprocessor = 82,
  cudaDevAttrManagedMemory = 83,
  cudaDevAttrIsMultiGpuBoard = 84,
  cudaDevAttrMultiGpuBoardGroupID = 85,
  cudaDevAttrHostNativeAtomicSupported = 86,
  cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
  cudaDevAttrPageableMemoryAccess = 88,
  cudaDevAttrConcurrentManagedAccess = 89,
  cudaDevAttrComputePreemptionSupported = 90,
  cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
  cudaDevAttrReserved92 = 92,
  cudaDevAttrReserved93 = 93,
  cudaDevAttrReserved94 = 94,
  cudaDevAttrCooperativeLaunch = 95,
  cudaDevAttrCooperativeMultiDeviceLaunch = 96,
  cudaDevAttrMaxSharedMemoryPerBlockOptin = 97
};

enum cudaFuncCache {
  cudaFuncCachePreferNone = 0,
  cudaFuncCachePreferShared = 1,
  cudaFuncCachePreferL1 = 2,
  cudaFuncCachePreferEqual = 3
};

__host__ cudaError_t cudaChooseDevice(int *device, const cudaDeviceProp *prop);
__host__ __device__ cudaError_t cudaDeviceGetAttribute(int *value,
                                                       cudaDeviceAttr attr,
                                                       int device);
__host__ cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);
__host__ __device__ cudaError_t
cudaDeviceGetCacheConfig(cudaFuncCache *pCacheConfig);
__host__ __device__ cudaError_t cudaDeviceGetLimit(size_t *pValue,
                                                   cudaLimit limit);
__host__ cudaError_t cudaDeviceGetP2PAttribute(int *value,
                                               cudaDeviceP2PAttr attr,
                                               int srcDevice, int dstDevice);
__host__ cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);
__host__ __device__ cudaError_t
cudaDeviceGetSharedMemConfig(cudaSharedMemConfig *pConfig);
__host__ cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                                      int *greatestPriority);
__host__ cudaError_t cudaDeviceReset(void);
__host__ cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);
__host__ cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
__host__ cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);
__host__ __device__ cudaError_t cudaDeviceSynchronize(void);
__host__ __device__ cudaError_t cudaGetDevice(int *device);
__host__ __device__ cudaError_t cudaGetDeviceCount(int *count);
__host__ cudaError_t cudaGetDeviceFlags(unsigned int *flags);
__host__ cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
__host__ cudaError_t cudaIpcCloseMemHandle(void *devPtr);
__host__ cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,
                                           cudaEvent_t event);
__host__ cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle,
                                         void *devPtr);
__host__ cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event,
                                            cudaIpcEventHandle_t handle);
__host__ cudaError_t cudaIpcOpenMemHandle(void **devPtr,
                                          cudaIpcMemHandle_t handle,
                                          unsigned int flags);
__host__ cudaError_t cudaSetDevice(int device);
__host__ cudaError_t cudaSetDeviceFlags(unsigned int flags);
__host__ cudaError_t cudaSetValidDevices(int *device_arr, int len);

// 4.2. Thread Management [DEPRECATED] (Not implemented)

//__host__ cudaError_t cudaThreadExit (void);
//__host__ cudaError_t cudaThreadGetCacheConfig(cudaFuncCache *pCacheConfig);
//__host__ cudaError_t cudaThreadGetLimit (size_t *pValue, cudaLimit limit);
//__host__ cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig);
//__host__ cudaError_t cudaThreadSetLimit (cudaLimit limit, size_t value);
//__host__ cudaError_t cudaThreadSynchronize (void);
//

// 4.3. Error Handling
__host__ __device__ const char *cudaGetErrorName(cudaError_t error);
__host__ __device__ const char *cudaGetErrorString(cudaError_t error);
__host__ __device__ cudaError_t cudaGetLastError(void);
__host__ __device__ cudaError_t cudaPeekAtLastError(void);

// 4.4 Stream Management
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status,
                                     void *userData);
__host__ cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                           cudaStreamCallback_t callback,
                                           void *userData, unsigned int flags);
__host__ cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr,
                                              size_t length,
                                              unsigned int flags);
__host__ cudaError_t cudaStreamCreate(cudaStream_t *pStream);
__host__ __device__ cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                                          unsigned int flags);
__host__ cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                                  unsigned int flags,
                                                  int priority);
__host__ __device__ cudaError_t cudaStreamDestroy(cudaStream_t stream);
__host__ cudaError_t cudaStreamGetFlags(cudaStream_t hStream,
                                        unsigned int *flags);
__host__ cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority);
__host__ cudaError_t cudaStreamQuery(cudaStream_t stream);
__host__ cudaError_t cudaStreamSynchronize(cudaStream_t stream);
__host__ __device__ cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
                                                    cudaEvent_t event,
                                                    unsigned int flags);

// 4.5 Event Management
__host__ cudaError_t cudaEventCreate(cudaEvent_t *event);
__host__ __device__ cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
                                                         unsigned int flags);
__host__ __device__ cudaError_t cudaEventDestroy(cudaEvent_t event);
__host__ cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                          cudaEvent_t end);
__host__ cudaError_t cudaEventQuery(cudaEvent_t event);
__host__ __device__ cudaError_t cudaEventRecord(cudaEvent_t event,
                                                cudaStream_t stream);
__host__ cudaError_t cudaEventSynchronize(cudaEvent_t event);

// 4.6 Execution Control
__host__ __device__ cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr,
                                                      const void *func);
__host__ cudaError_t cudaFuncSetAttribute(const void *func,
                                          cudaFuncAttribute attr, int value);
__host__ cudaError_t cudaFuncSetCacheConfig(const void *func,
                                            cudaFuncCache cacheConfig);
__host__ cudaError_t cudaFuncSetSharedMemConfig(const void *func,
                                                cudaSharedMemConfig config);
__device__ void *cudaGetParameterBuffer(size_t alignment, size_t size);
__device__ void *cudaGetParameterBufferV2(void *func, dim3 gridDimension,
                                          dim3 blockDimension,
                                          unsigned int sharedMemSize);
__host__ cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim,
                                                 dim3 blockDim, void **args,
                                                 size_t sharedMem,
                                                 cudaStream_t stream);
__host__ cudaError_t cudaLaunchCooperativeKernelMultiDevice(
    cudaLaunchParams *launchParamsList, unsigned int numDevices,
    unsigned int flags);
__host__ cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
                                      dim3 blockDim, void **args,
                                      size_t sharedMem, cudaStream_t stream);
__host__ cudaError_t cudaSetDoubleForDevice(double *d);
__host__ cudaError_t cudaSetDoubleForHost(double *d);

// 4.7 Occupancy
__host__ __device__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
__host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags);

// 4.8 Execution Control [Deprecated] (Not Implemented)
__host__ cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                       size_t sharedMem = 0,
                                       cudaStream_t stream = 0);
__host__ cudaError_t cudaLaunch(const void *func);
__host__ cudaError_t cudaSetupArgument(const void *arg, size_t size,
                                       size_t offset);

// 4.9 Memory Management
__host__ cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc *desc,
                                      cudaExtent *extent, unsigned int *flags,
                                      cudaArray_t array);
__host__ __device__ cudaError_t cudaFree(void *devPtr);
__host__ cudaError_t cudaFreeArray(cudaArray_t array);
__host__ cudaError_t cudaFreeHost(void *ptr);
__host__ cudaError_t
cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
__host__ cudaError_t cudaGetMipmappedArrayLevel(
    cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray,
    unsigned int level);
__host__ cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol);
__host__ cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol);
__host__ cudaError_t cudaHostAlloc(void **pHost, size_t size,
                                   unsigned int flags);
__host__ cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                              unsigned int flags);
__host__ cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost);
__host__ cudaError_t cudaHostRegister(void *ptr, size_t size,
                                      unsigned int flags);
__host__ cudaError_t cudaHostUnregister(void *ptr);
__host__ __device__ cudaError_t cudaMalloc(void **devPtr, size_t size);
__host__ cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
                                  cudaExtent extent);
__host__ cudaError_t cudaMalloc3DArray(cudaArray_t *array,
                                       const cudaChannelFormatDesc *desc,
                                       cudaExtent extent, unsigned int flags);
__host__ cudaError_t cudaMallocArray(cudaArray_t *array,
                                     const cudaChannelFormatDesc *desc,
                                     size_t width, size_t height,
                                     unsigned int flags);
__host__ cudaError_t cudaMallocHost(void **ptr, size_t size);
__host__ cudaError_t cudaMallocManaged(void **devPtr, size_t size,
                                       unsigned int flags);
__host__ cudaError_t cudaMallocMipmappedArray(
    cudaMipmappedArray_t *mipmappedArray, const cudaChannelFormatDesc *desc,
    cudaExtent extent, unsigned int numLevels, unsigned int flags);
__host__ cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width,
                                     size_t height);
__host__ cudaError_t cudaMemAdvise(const void *devPtr, size_t count,
                                   cudaMemoryAdvise advice, int device);
__host__ cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                                cudaMemcpyKind kind);
__host__ cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                                  size_t spitch, size_t width, size_t height,
                                  cudaMemcpyKind kind);
__host__ cudaError_t cudaMemcpy2DArrayToArray(
    cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
    cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width,
    size_t height, cudaMemcpyKind kind);
__host__ __device__ cudaError_t cudaMemcpy2DAsync(
    void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
    size_t height, cudaMemcpyKind kind, cudaStream_t stream);
__host__ cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
                                           cudaArray_const_t src,
                                           size_t wOffset, size_t hOffset,
                                           size_t width, size_t height,
                                           cudaMemcpyKind kind);
__host__ cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
                                                cudaArray_const_t src,
                                                size_t wOffset, size_t hOffset,
                                                size_t width, size_t height,
                                                cudaMemcpyKind kind,
                                                cudaStream_t stream);
__host__ cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset,
                                         size_t hOffset, const void *src,
                                         size_t spitch, size_t width,
                                         size_t height, cudaMemcpyKind kind);
__host__ cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset,
                                              size_t hOffset, const void *src,
                                              size_t spitch, size_t width,
                                              size_t height,
                                              cudaMemcpyKind kind,
                                              cudaStream_t stream);
__host__ cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *p);
__host__ __device__ cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *p,
                                                  cudaStream_t stream);
__host__ cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms *p);
__host__ cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms *p,
                                           cudaStream_t stream);
__host__ cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst,
                                            size_t hOffsetDst,
                                            cudaArray_const_t src,
                                            size_t wOffsetSrc,
                                            size_t hOffsetSrc, size_t count,
                                            cudaMemcpyKind kind);
__host__ __device__ cudaError_t cudaMemcpyAsync(void *dst, const void *src,
                                                size_t count,
                                                cudaMemcpyKind kind,
                                                cudaStream_t stream);
__host__ cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src,
                                         size_t wOffset, size_t hOffset,
                                         size_t count, cudaMemcpyKind kind);
__host__ cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src,
                                              size_t wOffset, size_t hOffset,
                                              size_t count, cudaMemcpyKind kind,
                                              cudaStream_t stream);
__host__ cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                          size_t count, size_t offset,
                                          cudaMemcpyKind kind);
__host__ cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                               size_t count, size_t offset,
                                               cudaMemcpyKind kind,
                                               cudaStream_t stream);
__host__ cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src,
                                    int srcDevice, size_t count);
__host__ cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
                                         const void *src, int srcDevice,
                                         size_t count, cudaStream_t stream);
__host__ cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset,
                                       size_t hOffset, const void *src,
                                       size_t count, cudaMemcpyKind kind);
__host__ cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset,
                                            size_t hOffset, const void *src,
                                            size_t count, cudaMemcpyKind kind,
                                            cudaStream_t stream);
__host__ cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                                        size_t count, size_t offset,
                                        cudaMemcpyKind kind);
__host__ cudaError_t cudaMemcpyToSymbolAsync(const void *symbol,
                                             const void *src, size_t count,
                                             size_t offset, cudaMemcpyKind kind,
                                             cudaStream_t stream);
__host__ cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
__host__ cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
                                          int dstDevice, cudaStream_t stream);
__host__ cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize,
                                              cudaMemRangeAttribute attribute,
                                              const void *devPtr, size_t count);
__host__ cudaError_t cudaMemRangeGetAttributes(
    void **data, size_t *dataSizes, cudaMemRangeAttribute *attributes,
    size_t numAttributes, const void *devPtr, size_t count);
__host__ cudaError_t cudaMemset(void *devPtr, int value, size_t count);
__host__ cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value,
                                  size_t width, size_t height);
__host__ __device__ cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch,
                                                  int value, size_t width,
                                                  size_t height,
                                                  cudaStream_t stream);
__host__ cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value,
                                  cudaExtent extent);
__host__ __device__ cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr,
                                                  int value, cudaExtent extent,
                                                  cudaStream_t stream);
__host__ __device__ cudaError_t cudaMemsetAsync(void *devPtr, int value,
                                                size_t count,
                                                cudaStream_t stream);
__host__ cudaExtent make_cudaExtent(size_t w, size_t h, size_t d);
__host__ cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz,
                                            size_t ysz);
__host__ cudaPos make_cudaPos(size_t x, size_t y, size_t z);

// 4.10. Unified Addressing

__host__ cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                              const void *ptr);

// 4.11. Peer Device Memory Access

__host__ cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
                                             int peerDevice);
__host__ cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
__host__ cudaError_t cudaDeviceEnablePeerAccess(int peerDevice,
                                                unsigned int flags);

// 4.12. OpenGL Interoperability

enum cudaGLDeviceList {
  cudaGLDeviceListAll = 1,
  cudaGLDeviceListCurrentFrame = 2,
  cudaGLDeviceListNextFrame = 3
};

__host__ cudaError_t cudaGLGetDevices(unsigned int *pCudaDeviceCount,
                                      int *pCudaDevices,
                                      unsigned int cudaDeviceCount,
                                      cudaGLDeviceList deviceList);
//__host__ cudaError_t cudaGraphicsGLRegisterBuffer (cudaGraphicsResource
//**resource, GLuint buffer, unsigned int flags);
//__host__ cudaError_t cudaGraphicsGLRegisterImage (cudaGraphicsResource
//**resource, GLuint image, GLenum target, unsigned int flags);
//__host__ cudaError_t cudaWGLGetDevice (int *device, HGPUNV hGpu);

// 4.13. OpenGL Interoperability [DEPRECATED] (Not implemented)
// enum cudaGLMapFlags {
// cudaGLMapFlagsNone = 0,
// cudaGLMapFlagsReadOnly = 1,
// cudaGLMapFlagsWriteDiscard = 2
//};

//__host__ cudaError_t cudaGLMapBufferObject (void **devPtr, GLuint bufObj);
//__host__ cudaError_t cudaGLMapBufferObjectAsync(void **devPtr, GLuint bufObj,
// cudaStream_t stream);
//__host__ cudaError_t cudaGLRegisterBufferObject(GLuint bufObj);
//__host__ cudaError_t cudaGLSetBufferObjectMapFlags(GLuint bufObj, unsigned int
// flags);
//__host__ cudaError_t cudaGLSetGLDevice (int device);
//__host__ cudaError_t cudaGLUnmapBufferObject (GLuint bufObj);
//__host__ cudaError_t cudaGLUnmapBufferObjectAsync(GLuint bufObj, cudaStream_t
// stream);
//__host__ cudaError_t cudaGLUnregisterBufferObject(GLuint bufObj);

// 4.14. Direct3D 9 Interoperability
enum cudaD3D9DeviceList {
  cudaD3D9DeviceListAll = 1,
  cudaD3D9DeviceListCurrentFrame = 2,
  cudaD3D9DeviceListNextFrame = 3
};

//__host__ cudaError_t cudaD3D9GetDevice (int *device, const char
//*pszAdapterName);
//__host__ cudaError_t cudaD3D9GetDevices (unsigned int *pCudaDeviceCount, int
//*pCudaDevices, unsigned int cudaDeviceCount, IDirect3DDevice9 *pD3D9Device,
// cudaD3D9DeviceList deviceList);
//__host__ cudaError_t cudaD3D9GetDirect3DDevice(IDirect3DDevice9
//**ppD3D9Device);
//__host__ cudaError_t cudaD3D9SetDirect3DDevice(IDirect3DDevice9 *pD3D9Device,
// int device);
//__host__ cudaError_t cudaGraphicsD3D9RegisterResource(cudaGraphicsResource
//**resource, IDirect3DResource9 *pD3DResource, unsigned int flags);

// 4.15. Direct3D 9 Interoperability [DEPRECATED] (Not implemented)
// enum cudaD3D9MapFlags {
// cudaD3D9MapFlagsNone = 0
// cudaD3D9MapFlagsReadOnly = 1,
// cudaD3D9MapFlagsWriteDiscard = 2
//};
// enum cudaD3D9RegisterFlags {
// cudaD3D9RegisterFlagsNone = 0,
// cudaD3D9RegisterFlagsArray = 1
//};
//__host__ cudaError_t cudaD3D9MapResources (int count, IDirect3DResource9
//**ppResources);
//__host__ cudaError_t cudaD3D9RegisterResource (IDirect3DResource9 *pResource,
// unsigned int flags);
//__host__ cudaError_t cudaD3D9ResourceGetMappedArray (cudaArray **ppArray,
// IDirect3DResource9 *pResource, unsigned int face, unsigned int level);
//__host__ cudaError_t cudaD3D9ResourceGetMappedPitch (size_t *pPitch, size_t
//*pPitchSlice, IDirect3DResource9 *pResource, unsigned int face, unsigned int
// level);
//__host__ cudaError_t cudaD3D9ResourceGetMappedPointer (void **pPointer,
// IDirect3DResource9 *pResource, unsigned int face, unsigned int level);
//__host__ cudaError_t cudaD3D9ResourceGetMappedSize (size_t *pSize,
// IDirect3DResource9 *pResource, unsigned int face, unsigned int level);
//__host__ cudaError_t cudaD3D9ResourceGetSurfaceDimensions (size_t *pWidth,
// size_t *pHeight, size_t *pDepth, IDirect3DResource9 *pResource, unsigned int
// face, unsigned int level);
//__host__ cudaError_t cudaD3D9ResourceSetMapFlags(IDirect3DResource9
//*pResource, unsigned int flags);
//__host__ cudaError_t cudaD3D9UnmapResources (int count, IDirect3DResource9
//**ppResources);
//__host__ cudaError_t cudaD3D9UnregisterResource(IDirect3DResource9
//*pResource);

// 4.16. Direct3D 10 Interoperability
// TODO

// 4.17. Direct3D 10 Interoperability [DEPRECATED] (not implemented)
// TODO

// 4.18. Direct3D 11 Interoperability
// TODO

// 4.19. Direct3D 11 Interoperability [DEPRECATED]
// TODO

// 4.20. VDPAU Interoperability
// TODO

// 4.21. EGL Interoperability
// TODO

// 4.22. Graphics Interoperability
// TODO

// 4.23. Texture Reference Management
// TODO

// 4.24. Surface Reference Management
// TODO

// 4.25. Texture Object Management
// TODO

// 4.26. Surface Object Management
// TODO

// 4.27. Version Management
__host__ cudaError_t cudaDriverGetVersion(int *driverVersion);
__host__ __device__ cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);

// 4.28. C++ API Routines
#if defined(__cplusplus)

#endif
// TODO

// 4.29. Interactions with the CUDA Driver API
// 4.30. Profiler Control
__host__ cudaError_t cudaProfilerInitialize(const char *configFile,
                                            const char *outputFile,
                                            cudaOutputMode_t outputMode);
__host__ cudaError_t cudaProfilerStart(void);
__host__ cudaError_t cudaProfilerStop(void);

#endif

#if defined(__cplusplus)
}
#endif
