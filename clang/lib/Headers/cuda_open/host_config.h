//===--------------- cuda_open/host_config.h  -----------------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef __CUDA_OPEN_HOST_CONFIG_H__
#define __CUDA_OPEN_HOST_CONFIG_H__
#ifndef __CUDA__
#error "This file is for CUDA compilation only."
#endif
#ifndef __USE_OPEN_HEADERS__
#error "This file requires -D__USE_OPEN_HEADERS__"
#endif

#endif
