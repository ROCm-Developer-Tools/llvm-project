//===-- offload-arch/OffloadArch.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file  amdgpu/vendor_specific_capabilities.cpp
///
/// Implementiton of _aot_amdgpu_capabilities() function for offload-arch tool.
/// This is only called with the -r flag to show all runtime capabilities that
/// would satisfy requirements of the compiled image.
///
//===----------------------------------------------------------------------===//

#include "../OffloadArch.h"

// So offload-arch can be built without hsa installed a copy of hsa.h
// is stored with the tool in the vendor specific directory.  This combined
// with dynamic loading (at runtime) of "libhsa-runtime64.so" allows
// offload-arch to be built without hsa installed.  Of course hsa
// (rocr runtime) must be operational at runtime.
//
#include "hsa-subset.h"
#include <dlfcn.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <vector>

struct amdgpu_features_t {
  char *name_str;
  uint32_t workgroup_max_size;
  hsa_dim3_t grid_max_dim;
  uint64_t grid_max_size;
  uint32_t fbarrier_max_size;
  uint16_t workgroup_max_dim[3];
  bool def_rounding_modes[3];
  bool base_rounding_modes[3];
  bool mach_models[2];
  bool profiles[2];
  bool fast_f16;
};

// static pointers to dynamically loaded HSA functions used in this module.
static hsa_status_t (*_dl_hsa_init)();
static hsa_status_t (*_dl_hsa_shut_down)();
static hsa_status_t (*_dl_hsa_isa_get_info_alt)(hsa_isa_t, hsa_isa_info_t,
                                                void *);
static hsa_status_t (*_dl_hsa_agent_get_info)(hsa_agent_t, hsa_agent_info_t,
                                              void *);
static hsa_status_t (*_dl_hsa_iterate_agents)(
    hsa_status_t (*callback)(hsa_agent_t, void *), void *);
static hsa_status_t (*_dl_hsa_agent_iterate_isas)(
    hsa_agent_t, hsa_status_t (*callback)(hsa_isa_t, void *), void *);

// These two static vectors are created by HSA iterators and needed after
// iterators complete, so we save them statically.
static std::vector<amdgpu_features_t> AMDGPU_FEATUREs;
static std::vector<hsa_agent_t *> HSA_AGENTs;

static std::string offload_arch_requested;
static bool first_call = true;

#define _return_on_err(err)                                                    \
  {                                                                            \
    if ((err) != HSA_STATUS_SUCCESS) {                                         \
      return (err);                                                            \
    }                                                                          \
  }

static hsa_status_t get_isa_info(hsa_isa_t isa, void *data) {
  hsa_status_t err;
  amdgpu_features_t isa_i;
  int *isa_int = reinterpret_cast<int *>(data);
  (*isa_int)++;

  std::string isa_str("ISA ");
  isa_str += std::to_string(*isa_int);

  uint32_t name_len;
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &name_len);
  _return_on_err(err);
  isa_i.name_str = new char[name_len];
  if (isa_i.name_str == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_i.name_str);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_MACHINE_MODELS,
                                 isa_i.mach_models);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_PROFILES, isa_i.profiles);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES,
                                 isa_i.def_rounding_modes);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(
      isa, HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES,
      isa_i.base_rounding_modes);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_FAST_F16_OPERATION,
                                 &isa_i.fast_f16);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_WORKGROUP_MAX_DIM,
                                 &isa_i.workgroup_max_dim);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_WORKGROUP_MAX_SIZE,
                                 &isa_i.workgroup_max_size);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_GRID_MAX_DIM,
                                 &isa_i.grid_max_dim);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_GRID_MAX_SIZE,
                                 &isa_i.grid_max_size);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_FBARRIER_MAX_SIZE,
                                 &isa_i.fbarrier_max_size);
  _return_on_err(err);

  AMDGPU_FEATUREs.push_back(isa_i);
  return err;
}

static hsa_status_t iterateAgentsCallback(hsa_agent_t Agent, void *Data) {
  hsa_device_type_t DeviceType;
  hsa_status_t Status =
      _dl_hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DeviceType);

  // continue only if device type if GPU
  if (Status != HSA_STATUS_SUCCESS || DeviceType != HSA_DEVICE_TYPE_GPU) {
    return Status;
  }

  std::vector<std::string> *GPUs =
      static_cast<std::vector<std::string> *>(Data);
  char GPUName[64];
  Status = _dl_hsa_agent_get_info(Agent, HSA_AGENT_INFO_NAME, GPUName);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  if (offload_arch_requested.compare(GPUName) == 0) {
    GPUs->push_back(GPUName);
    HSA_AGENTs.push_back(&Agent);
  }
  return HSA_STATUS_SUCCESS;
}

void *_aot_dynload_hsa_runtime() {

  const char *hsa_runtime_locations[] = {
      "/usr/lib/aomp/lib/libhsa-runtime64.so",
      "/opt/rocm/hsa/lib/libhsa-runtime64.so",
      "/opt/rocm-4.1.0/hsa/lib/libhsa-runtime64.so",
  };

  void *dlhandle = nullptr;
  // FIXME: before going through possible hsa locations try
  //        <directory-of-binary>/../lib/libhsa-runtime64.so

  struct stat stat_buffer;
  
  // First search in system library paths. Allows user to dynamically
  // load desired version of hsa runtime.
  dlhandle = dlopen("libhsa-runtime64.so", RTLD_NOW);

  // In case of failure, search in known absolute locations.
  if(!dlhandle) {
    for (auto *rt_loc : hsa_runtime_locations) {
      if (stat(rt_loc, &stat_buffer) == 0) {
        dlhandle = dlopen(rt_loc, RTLD_NOW);
        break;
      }
    }
  }

  // Return null if hsa runtime is not found in system paths and in
  // absolute locations.
  if (!dlhandle)
    return nullptr;

  // We could use real names of hsa functions but the _dl_ makes it clear
  // these are dynamically loaded
  *(void **)&_dl_hsa_init = dlsym(dlhandle, "hsa_init");
  *(void **)&_dl_hsa_shut_down = dlsym(dlhandle, "hsa_shut_down");
  *(void **)&_dl_hsa_isa_get_info_alt = dlsym(dlhandle, "hsa_isa_get_info_alt");
  *(void **)&_dl_hsa_agent_get_info = dlsym(dlhandle, "hsa_agent_get_info");
  *(void **)&_dl_hsa_iterate_agents = dlsym(dlhandle, "hsa_iterate_agents");
  *(void **)&_dl_hsa_agent_iterate_isas =
      dlsym(dlhandle, "hsa_agent_iterate_isas");
  return dlhandle;
}

std::string _aot_amdgpu_capabilities(uint16_t vid, uint16_t devid,
                                     std::string oa) {
  std::string amdgpu_capabilities;
  offload_arch_requested = oa;

  if (first_call) {
    first_call = false;
    void *dlhandle = _aot_dynload_hsa_runtime();
    if (!dlhandle) {
      amdgpu_capabilities.append(" HSAERROR-LOADING");
      return amdgpu_capabilities;
    }
    hsa_status_t Status = _dl_hsa_init();
    if (Status != HSA_STATUS_SUCCESS) {
      amdgpu_capabilities.append(" HSAERROR-INITIALIZATION");
      return amdgpu_capabilities;
    }
  }
  std::vector<std::string> GPUs;
  hsa_status_t Status = _dl_hsa_iterate_agents(iterateAgentsCallback, &GPUs);
  if (Status != HSA_STATUS_SUCCESS) {
    amdgpu_capabilities.append(" HSAERROR-AGENT_ITERATION");
    return amdgpu_capabilities;
  }
  if (GPUs.size() == 0) {
    amdgpu_capabilities.append("NOT-HSA-VISIBLE");
    return amdgpu_capabilities;
  }

  int isa_number = 0;
  hsa_agent_t *agent_ptr = HSA_AGENTs[isa_number];
  Status = _dl_hsa_agent_iterate_isas(*agent_ptr, get_isa_info, &isa_number);
  if (Status == HSA_STATUS_ERROR_INVALID_AGENT) {
    amdgpu_capabilities.append(" HSAERROR-INVALID_AGENT");
    return amdgpu_capabilities;
  }

  // parse features from field name_str of last amdgpu_features_t found
  std::string features(AMDGPU_FEATUREs[isa_number - 1].name_str);
  std::string::size_type prev_pos = 0, pos = 0;
  int fnum = 0;
  while ((pos = features.find(":", pos)) != std::string::npos) {
    std::string substring(features.substr(prev_pos, pos - prev_pos));
    if (fnum) {
      amdgpu_capabilities.append(" ");
      amdgpu_capabilities.append(substring);
    }
    prev_pos = ++pos;
    fnum++;
  }
  if (prev_pos) {
    amdgpu_capabilities.append(" ");
    amdgpu_capabilities.append(features.substr(prev_pos, pos - prev_pos));
  }
  // We cannot shutdown hsa or close dlhandle because
  // _aot_amd_capabilities could be called multiple times.
  return amdgpu_capabilities;
}
