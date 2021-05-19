//===-- offload-arch/OffloadArch.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of offload-arch tool and alias commands "amdgpu-arch" ,
/// "nvidia-arch" and "intelhd-arch". The alias commands are symbolic links
/// to offload-arch.
/// offload-arch prints the offload-arch for the current active system or
/// looks up numeric pci ids and codenames for a given offload-arch.
///
//===----------------------------------------------------------------------===//

#include "OffloadArch.h"
#include <stdio.h>

void aot_usage() {
  printf("\n\
   offload-arch: Print offload architecture(s) for current system, or\n\
                 print offload runtime capabilities of current system,\n\
                 or lookup information about offload architectures,\n\
                 or print offload requirements for an application binary\n\
\n\
   Usage:\n\
\n\
     offload-arch [ Options ] [ Optional lookup-value ]\n\
\n\
     With no options, offload-arch prints a value for the first offload arch\n\
     found in the current active system. This can be used by various clang\n\
     frontends. For example, to compile for openmp offloading on your current\n\
     system, invoke clang with the following command:\n\
\n\
        clang -fopenmp --offload-arch=`offload-arch` foo.c\n\
\n\
     If an optional lookup-value is specified, offload-arch will\n\
     check if the value is either a valid offload-arch or a codename\n\
     and lookup requested additional information. For example,\n\
     this provides all information for offload-arch gfx906:\n\
\n\
        offload-arch gfx906 -v \n\
\n\
     Options:\n\
     -h  Print this help message\n\
     -a  Print values for all devices. Don't stop at first device found.\n\
     -m  Print device code name (often found in pci.ids file)\n\
     -n  Print numeric pci-id\n\
     -t  Print clang offload triple to use for the offload arch.\n\
     -v  Verbose = -a -m -n -t  \n\
         For all devices, print codename, numeric value and triple\n\
     -f  <filename> Print offload requirements including offload-arch for\n\
         each compiled offload image built into an application binary file.\n\
	 This only works if the binary is not stripped of symbols.\n\
     -c  Print offload capabilities of the current active system.\n\
	 This option is used by the language runtime to select an image\n\
	 when multiple images are availble.\n\
	 A capability must exist for each requirement of the selected image.\n\
\n\
     There are aliases 'amdgpu-arch', 'nvidia-arch', and 'intelhd-arch' which\n\
     are created with symbolic links to offload-arch. These aliases return 1\n\
     if no amdgcn GPU, no cuda GPU or no intel HD GPU is found, respectfully.\n\
     These aliases are useful to gaurd the execution of architecture-specific\n\
     tests or to conditionally load archecture-specific software.\n\
\n\
     Copyright (c) 2021 ADVANCED MICRO DEVICES, INC.\n\
\n\
");
  exit(1);
}
