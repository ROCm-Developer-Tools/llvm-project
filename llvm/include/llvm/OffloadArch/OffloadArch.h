//==-- llvm/OffloadArch/OffloadArch.h --------------------------------*- C++
//-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains headers to use libLLVMOffloadArch.a
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOADARCH_OFFLOADARCH_H
#define LLVM_OFFLOADARCH_OFFLOADARCH_H

#include <iostream>
#include <string>
#include <vector>

#define MAXPATHSIZE 512

// These search phrases in /sys/bus/pci/devices/*/uevent are found even if
// the device driver is not running.
#define AMDGPU_SEARCH_PHRASE "DRIVER=amdgpu"
#define AMDGPU_PCIID_PHRASE "PCI_ID=1002:"
#define NVIDIA_SEARCH_PHRASE "DRIVER=nvidia"
#define NVIDIA_PCIID_PHRASE "PCI_ID=10DE:"
#define INTELHD_SEARCH_PHRASE "DRIVER=i915"
#define INTELHD_PCIID_PHRASE "PCI_ID=8086:"

///
/// Called by libomptarget runtime to get runtime capabilities.
extern "C" int
__aot_get_capabilities_for_runtime(char *offload_arch_output_buffer,
                                   size_t offload_arch_output_buffer_size);

/// Get the vendor specified softeare capabilities of the current runtime
/// The input vendor id selects the vendor function to call.
std::string _aot_get_capabilities(uint16_t vid, uint16_t devid, std::string oa);

/// Get the AMD specific software capabilities of the current runtime
std::string _aot_amdgpu_capabilities(uint16_t vid, uint16_t devid,
                                     std::string oa);
/// Get the Nvidia specific software capabilities of the current runtime
std::string _aot_nvidia_capabilities(uint16_t vid, uint16_t devid,
                                     std::string oa);
/// Get the Intel specific software capabilities of the current runtime
std::string _aot_intelhd_capabilities(uint16_t vid, uint16_t devid,
                                      std::string oa);

///  return requirements for each offload image in an application binary
std::vector<std::string> _aot_get_requirements_from_file(const std::string &fn);

///  return all offloadable pci-ids found in the system
std::vector<std::string> _aot_get_all_pci_ids();
///  return all offloadable pci-ids for a given vendor
std::vector<std::string> _aot_get_pci_ids(const char *driver_search_phrase,
                                          const char *pci_id_search_phrase);

///  lookup function to return all pci-ids for an input codename
std::vector<std::string> _aot_lookup_codename(std::string lookup_codename);

///  lookup function to return all pci-ids for an input offload_arch
std::vector<std::string>
_aot_lookup_offload_arch(std::string lookup_offload_arch);

/// get the offload arch for VendorId-DeviceId
std::string _aot_get_offload_arch(uint16_t VendorID, uint16_t DeviceID);

/// get the vendor specified codename VendorId-DeviceId
std::string _aot_get_codename(uint16_t VendorID, uint16_t DeviceID);

/// get the compilation triple for VendorId-DeviceId
std::string _aot_get_triple(uint16_t VendorID, uint16_t DeviceID);

/// Utility to return contents of a file as a string
std::string _aot_get_file_contents(std::string fname);

#endif
