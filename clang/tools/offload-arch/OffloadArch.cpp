//===-- offload-arch/OffloadArch.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of offload-arch tool and alias commands "amdgpu-arch" and
/// "nvidia-arch". The alias commands are symbolic links to offload-arch.
/// offload-arch prints the offload-arch for the current active system or
/// looks up numeric pci ids and codenames for a given offload-arch.
///
//===----------------------------------------------------------------------===//

#define __OFFLOAD_ARCH_MAIN__
#include "OffloadArch.h"
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#define MAXPATHSIZE 512

// These search phrases in /sys/bus/pci/devices/*/uevent are found even if
// the device driver is not running.
#define AMDGPU_SEARCH_PHRASE "DRIVER=amdgpu"
#define AMDGPU_PCIID_PHRASE "PCI_ID=1002:"
#define NVIDIA_SEARCH_PHRASE "DRIVER=nvidia"
#define NVIDIA_PCIID_PHRASE "PCI_ID=10DE:"

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
        clang -fopenmp -fopenmp-targets=`offload-arch` foo.c\n\
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
     -c  Print offload capabilities of the current active system.\n\
	 This option is used by the language runtime to select an image\n\
	 when multiple images are availble.\n\
	 A capability must exist for each requirement of the selected image.\n\
\n\
     There are symbolic link aliases 'amdgpu-arch' and 'nvidia-arch' for\n\
     offload-arch. These aliases return 1 if no amdgcn GPU or no cuda GPU\n\
     is found, respectfully. These aliases are useful to determine if\n\
     architecture-specific tests should be run, or to conditionally load\n\
     archecture-specific software.\n\
\n\
     Copyright (c) 2021 ADVANCED MICRO DEVICES, INC.\n\
\n\
");
  exit(1);
}

static bool AOT_get_all_active_devices;

std::string _aot_get_file_contents(std::string fname) {
  std::string file_contents;
  std::string line;
  std::ifstream myfile(fname);
  if (myfile.is_open()) {
    while (getline(myfile, line)) {
      file_contents.append(line).append("\n");
    }
    myfile.close();
  }
  return file_contents;
}

std::vector<std::string> _aot_get_pci_ids(const char *driver_search_phrase,
                                          const char *pci_id_search_phrase) {
  std::vector<std::string> PCI_IDS;
  char uevent_filename[MAXPATHSIZE];
  const char *sys_bus_pci_devices_dir = "/sys/bus/pci/devices";
  DIR *dirp;
  struct dirent *dir;

  dirp = opendir(sys_bus_pci_devices_dir);
  if (dirp) {
    while ((dir = readdir(dirp)) != 0) {
      // foreach subdir look for uevent file
      if ((strcmp(dir->d_name, ".") == 0) || (strcmp(dir->d_name, "..") == 0))
        continue;
      snprintf(uevent_filename, MAXPATHSIZE, "%s/%s/uevent",
               sys_bus_pci_devices_dir, dir->d_name);
      std::string file_contents =
          _aot_get_file_contents(std::string(uevent_filename));
      if (!file_contents.empty()) {
        std::size_t found_loc = file_contents.find(driver_search_phrase);
        if (found_loc != std::string::npos) {
          found_loc = file_contents.find(pci_id_search_phrase);
          if (found_loc != std::string::npos)
            PCI_IDS.push_back(file_contents.substr(found_loc + 7, 9));
        }
      }
    } // end of foreach subdir
    closedir(dirp);
  } else {
    fprintf(stderr, "ERROR: failed to open directory %s.\n",
            sys_bus_pci_devices_dir);
    exit(1);
  }
  return PCI_IDS;
}

std::vector<std::string> _aot_lookup_codename(std::string lookup_codename) {
  std::vector<std::string> PCI_IDS;
  for (auto id2str : AOT_CODENAMES)
    if (lookup_codename.compare(id2str.codename) == 0)
      for (auto aot_table_entry : AOT_TABLE) {
        if (id2str.codename_id == aot_table_entry.codename_id) {
          uint16_t VendorID;
          uint16_t DeviceID;
          char pci_id[10];
          VendorID = aot_table_entry.vendorid;
          DeviceID = aot_table_entry.devid;
          snprintf(&pci_id[0], 10, "%x:%x", VendorID, DeviceID);
          PCI_IDS.push_back(std::string(&pci_id[0]));
        }
      }
  return PCI_IDS;
}

std::vector<std::string>
_aot_lookup_offload_arch(std::string lookup_offload_arch) {
  std::vector<std::string> PCI_IDS;
  for (auto id2str : AOT_OFFLOADARCHS)
    if (lookup_offload_arch.compare(id2str.offloadarch) == 0)
      for (auto aot_table_entry : AOT_TABLE) {
        if (id2str.offloadarch_id == aot_table_entry.offloadarch_id) {
          uint16_t VendorID;
          uint16_t DeviceID;
          char pci_id[10];
          VendorID = aot_table_entry.vendorid;
          DeviceID = aot_table_entry.devid;
          snprintf(&pci_id[0], 10, "%x:%x", VendorID, DeviceID);
          PCI_IDS.push_back(std::string(&pci_id[0]));
        }
      }
  return PCI_IDS;
}

std::string _aot_get_codename(uint16_t VendorID, uint16_t DeviceID) {
  std::string retval;
  for (auto aot_table_entry : AOT_TABLE) {
    if ((VendorID == aot_table_entry.vendorid) &&
        (DeviceID == aot_table_entry.devid))
      for (auto id2str : AOT_CODENAMES)
        if (id2str.codename_id == aot_table_entry.codename_id)
          return std::string(id2str.codename);
  }
  return retval;
}

std::string _aot_get_offload_arch(uint16_t VendorID, uint16_t DeviceID) {
  std::string retval;
  for (auto aot_table_entry : AOT_TABLE) {
    if ((VendorID == aot_table_entry.vendorid) &&
        (DeviceID == aot_table_entry.devid))
      for (auto id2str : AOT_OFFLOADARCHS)
        if (id2str.offloadarch_id == aot_table_entry.offloadarch_id)
          return std::string(id2str.offloadarch);
  }
  return retval;
}

std::string _aot_get_capabilities(uint16_t vid, uint16_t devid,
                                  std::string oa) {
  std::string capabilities(" ");
  switch (vid) {
  case 0x1002:
    capabilities.append(_aot_amdgpu_capabilities(vid, devid, oa));
    break;
  case 0x10de:
    capabilities.append(_aot_nvidia_capabilities(vid, devid, oa));
    break;
  }
  return capabilities;
}

std::string _aot_get_triple(uint16_t VendorID, uint16_t DeviceID) {
  std::string retval;
  switch (VendorID) {
  case 0x1002:
    return (std::string("amdgcn-amd-amdhsa"));
    break;
  case 0x10de:
    return (std::string("nvptx64-nvidia-cuda"));
    break;
  }
  return retval;
}

int main(int argc, char **argv) {
  bool print_codename = false;
  bool print_numeric = false;
  bool print_capabilities_for_runtime_requirements = false;
  bool amdgpu_arch = false;
  bool nvidia_arch = false;
  AOT_get_all_active_devices = false;
  bool print_triple = false;
  std::string lookup_value;
  std::string a, input_filename;
  for (int argi = 0; argi < argc; argi++) {
    a = std::string(argv[argi]);
    if (argi == 0) {
      // look for arch-specific invocation with symlink
      amdgpu_arch = (a.find("amdgpu-arch") != std::string::npos);
      nvidia_arch = (a.find("nvidia-arch") != std::string::npos);
    } else {
      if (a == "-n") {
        print_numeric = true;
      } else if (a == "-m") {
        print_codename = true;
      } else if (a == "-c") {
        print_capabilities_for_runtime_requirements = true;
      } else if (a == "-h") {
        aot_usage();
      } else if (a == "-a") {
        AOT_get_all_active_devices = true;
      } else if (a == "-t") {
        print_triple = true;
      } else if (a == "-v") {
        AOT_get_all_active_devices = true;
        print_codename = true;
        print_numeric = true;
        print_triple = true;
      } else if (a == "-f") {
        argi++;
        if (argi == argc) {
          fprintf(stderr, "ERROR: Missing filename for -f option\n");
          return 1;
        }
        input_filename = std::string(argv[argi]);
      } else {
        lookup_value = a;
      }
    }
  }

  std::vector<std::string> PCI_IDS;

  if (!input_filename.empty()) {

    PCI_IDS = _aot_get_requirements_from_file(input_filename);
    if (PCI_IDS.empty())
      return 1;
    for (auto PCI_ID : PCI_IDS)
      printf("%s\n", PCI_ID.c_str());
    return 0;

  } else if (lookup_value.empty()) {

    // No lookup_value so get the current pci ids.
    // First check if invocation was arch specific.
    if (amdgpu_arch) {
      PCI_IDS = _aot_get_pci_ids(AMDGPU_SEARCH_PHRASE, AMDGPU_PCIID_PHRASE);
    } else if (nvidia_arch) {
      PCI_IDS = _aot_get_pci_ids(NVIDIA_SEARCH_PHRASE, NVIDIA_PCIID_PHRASE);
    } else {
      // Search for all supported offload archs;
      PCI_IDS = _aot_get_pci_ids(AMDGPU_SEARCH_PHRASE, AMDGPU_PCIID_PHRASE);
      for (auto PCI_ID :
           _aot_get_pci_ids(NVIDIA_SEARCH_PHRASE, NVIDIA_PCIID_PHRASE))
        PCI_IDS.push_back(PCI_ID);
    }

  } else {
    if (print_capabilities_for_runtime_requirements) {
      fprintf(stderr, "ERROR: cannot lookup offload-arch/codename AND query\n");
      fprintf(stderr, "       active runtime capabilities (-c).\n");
      return 1;
    }
    PCI_IDS = _aot_lookup_offload_arch(lookup_value);
    if (PCI_IDS.empty())
      PCI_IDS = _aot_lookup_codename(lookup_value);
    if (PCI_IDS.empty()) {
      fprintf(stderr, "ERROR: Could not find \"%s\" in offload-arch tables\n",
              lookup_value.c_str());
      fprintf(stderr, "       as either an offload-arch or a codename.\n");
      return 1;
    }
  }

  if (PCI_IDS.empty()) {
    return 1;
  }

  int rc = 0;
  bool first_device_printed = false;
  for (auto PCI_ID : PCI_IDS) {
    unsigned vid32, devid32;
    sscanf(PCI_ID.c_str(), "%x:%x", &vid32, &devid32);
    uint16_t vid = vid32;
    uint16_t devid = devid32;
    std::string offload_arch = _aot_get_offload_arch(vid, devid);
    if (offload_arch.empty()) {
      fprintf(stderr, "ERROR: offload-arch not found for %x:%x.\n", vid, devid);
      rc = 1;
    } else {
      std::string xinfo;
      if (print_codename)
        xinfo.append(" ").append(_aot_get_codename(vid, devid));
      if (print_numeric)
        xinfo.append(" ").append(PCI_ID);
      if (print_triple)
        xinfo.append(" ").append(_aot_get_triple(vid, devid));
      if (print_capabilities_for_runtime_requirements) {
        xinfo.append(" ").append(
            _aot_get_capabilities(vid, devid, offload_arch));
      }
      if (AOT_get_all_active_devices || !first_device_printed) {
        first_device_printed = true;
        printf("%s%s\n", offload_arch.c_str(), xinfo.c_str());
      } else {
        break;
      }
    }
  }
  return rc;
}
