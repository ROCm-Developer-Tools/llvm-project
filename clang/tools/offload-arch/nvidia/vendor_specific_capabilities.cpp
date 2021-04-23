//===-- offload-arch/OffloadArch.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file  nvidia/vendor_specific_capabilities.cpp
///
/// Implementation of vendor specific runtime capabilities
///
//===----------------------------------------------------------------------===//

#include "../OffloadArch.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

std::string _aot_nvidia_capabilities(uint16_t vid, uint16_t devid,
                                     std::string oa) {
  std::string nvidia_capabilities;
  std::string file_contents =
      _aot_get_file_contents(std::string("/sys/module/nvidia/version"));
  if (!file_contents.empty()) {
    // parse nvidia kernel module version and release
    int ver, rel;
    char sbuf[16];
    sscanf(file_contents.c_str(), "%d.%d\n", &ver, &rel);
    snprintf(sbuf, 16, "v%d.%d", ver, rel);
    nvidia_capabilities.append(sbuf);
  }
  // FIXME: Add capability for running cuda version
  return nvidia_capabilities;
}
