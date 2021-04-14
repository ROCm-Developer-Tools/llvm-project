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
/// This tool prints offload architecture(s) for the current active system.\n\
///
//===----------------------------------------------------------------------===//

#include "pci_ids.h"
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include "generated_offload_arch.h"

#define MAXPATHSIZE 512

void aot_usage() {
  printf("\n\
   offload-arch: Print offload architecture(s) for the current active system.\n\
\n\
   Usage:\
\n\
     offload-arch [ Options ] [ Optional lookup-value ]\n\
\n\
     With no options, offload-arch prints a value for first offload architecture \n\
     found in current system.  This value can be used by various clang frontends.\n\
     For example, to compile for openmp offloading on current current system\n\
     one could invoke clang with the following command:\n\
\n\
     clang -fopenmp -openmp-targets=`offload-arch` foo.c\n\
\n\
     If an optional lookup-value is specified, offload-arch\n\
     will check if the value is either a valid offload-arch or codename\n\
     and lookup requested additional information. For example,\n\
     this provides the all information for offload-arch gfx906\n\
\n\
     offload-arch gfx906 -v \n\
\n\
     Options:\n\
     -h  Print this help message\n\
     -a  Print value for each device found. Don't stop at first device found.\n\
     -c  Print short codename instead of targetid\n\
     -n  Print numeric pci-id, useful to fix tables if getting unknown targetid\n\
     -t  Print the recommended offload triple.\n\
     -v  Verbose = -a -c -n -t \n\
     -l  Print long codename as found in pci.ids file\n\
     -r  Print capabilities of current system to satisfy potential runtime\n\
         requirements of compiled offload images.  This is used by runtime to\n\
	 choose correct image when multiple compiled images are availble.\n\
\n\
     The alias amdgpu-arch returns 1 if no amdgcn GPU is found.\n\
     The alias nvidia-arch returns 1 if no cuda GPU is found.\n\
     These aliases are useful to determine if architecture-specific tests\n\
     should be run. Or these aliases could be used to conditionally load\n\
     archecture-specific software at runtime.\n\
\n\
   Copyright (c) 2021 ADVANCED MICRO DEVICES, INC.\n\
\n\
");
  exit(1);
}

static bool AOT_get_all_active_devices;

std::vector<std::string> _aot_get_pci_ids(const char *driver_search_phrase) {
  std::vector<std::string> PCI_IDS;
  char uevent_filename[MAXPATHSIZE];
  const char *sys_bus_pci_devices_dir = "/sys/bus/pci/devices";
  DIR *dirp;
  struct dirent *dir;
  char *file_contents;
  size_t bytes_read;
  FILE *fd;

  dirp = opendir(sys_bus_pci_devices_dir);
  if (dirp) {
    while ((dir = readdir(dirp)) != 0) {
      // foreach subdir look for uevent file
      if ((strcmp(dir->d_name, ".") == 0) || (strcmp(dir->d_name, "..") == 0))
        continue;
      snprintf(uevent_filename, MAXPATHSIZE, "%s/%s/uevent",
               sys_bus_pci_devices_dir, dir->d_name);
      fd = fopen(uevent_filename, "r");
      if (fd) {
        fseek(fd, 0, SEEK_END);
        size_t fSize = ftell(fd);
        rewind(fd);
        file_contents = (char *)malloc(sizeof(char) * fSize);
        if (file_contents == NULL) {
          fprintf(stderr, "malloc fail for %ld bytes\n", fSize);
          exit(2);
        }
        bytes_read = fread(file_contents, 1, fSize, fd);
        if (bytes_read > fSize) {
          fprintf(stderr, "Read error on %s\n", uevent_filename);
          exit(3);
        }
        std::string str_contents(file_contents);
        std::size_t driver_found_loc = str_contents.find(driver_search_phrase);
        if (driver_found_loc == 0) {
          std::size_t pcipos = str_contents.find("PCI_ID") + 7;
          std::string remainder = str_contents.substr(pcipos);
          std::size_t nextval = remainder.find("PCI") - 1;
          PCI_IDS.push_back(remainder.substr(0, nextval));
          if (!AOT_get_all_active_devices) {
            free(file_contents);
            fclose(fd);
            return PCI_IDS;
          }
        }
        free(file_contents);
        fclose(fd);
      } else {
        fprintf(stderr, "File error opening %s\n", uevent_filename);
        exit(1);
      }
    } // end of foreach subdir
    closedir(dirp);
  } else {
    fprintf(stderr, "Could not open dir %s \n", sys_bus_pci_devices_dir);
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
          sprintf(&pci_id[0], "%x:%x", VendorID, DeviceID);
          PCI_IDS.push_back(std::string(&pci_id[0]));
          if (!AOT_get_all_active_devices)
            return PCI_IDS;
        }
      }
  return PCI_IDS;
}

std::vector<std::string>
_aot_lookup_offload_arch(std::string lookup_offload_arch) {
  std::vector<std::string> PCI_IDS;
  for (auto id2str : AOT_TARGETS)
    if (lookup_offload_arch.compare(id2str.target) == 0)
      for (auto aot_table_entry : AOT_TABLE) {
        if (id2str.target_id == aot_table_entry.target_id) {
          uint16_t VendorID;
          uint16_t DeviceID;
          char pci_id[10];
          VendorID = aot_table_entry.vendorid;
          DeviceID = aot_table_entry.devid;
          sprintf(&pci_id[0], "%x:%x", VendorID, DeviceID);
          PCI_IDS.push_back(std::string(&pci_id[0]));
          if (!AOT_get_all_active_devices)
            return PCI_IDS;
        }
      }
  return PCI_IDS;
}

std::string _aot_get_codename(uint16_t VendorID, uint16_t DeviceID) {
  for (auto aot_table_entry : AOT_TABLE) {
    if ((VendorID == aot_table_entry.vendorid) &&
        (DeviceID == aot_table_entry.devid))
      for (auto id2str : AOT_CODENAMES)
        if (id2str.codename_id == aot_table_entry.codename_id)
          return std::string(id2str.codename);
  }
  return nullptr;
}

std::string _aot_get_target(uint16_t VendorID, uint16_t DeviceID) {
  for (auto aot_table_entry : AOT_TABLE) {
    if ((VendorID == aot_table_entry.vendorid) &&
        (DeviceID == aot_table_entry.devid))
      for (auto id2str : AOT_TARGETS)
        if (id2str.target_id == aot_table_entry.target_id)
          return std::string(id2str.target);
  }
  return nullptr;
}

std::string _aot_get_amdgpu_capabilities() {
  std::string amdgpu_capabilities;
  char *file_contents;
  size_t bytes_read;
  FILE *fd;
  fd = fopen("/sys/module/amdgpu/version", "r");
  if (!fd) {
    fprintf(stderr, "Unable to open /sys/module/amdgpu/version\n");
    exit(1);
  }
  fseek(fd, 0, SEEK_END);
  size_t fSize = ftell(fd);
  rewind(fd);
  file_contents = (char *)malloc(sizeof(char) * fSize);
  bytes_read = fread(file_contents, 1, fSize, fd);
  if (bytes_read > fSize) {
    fprintf(stderr, "Read error on /sys/module/amdgpu/version\n");
    exit(3);
  }
  int ver, rel, mod;
  sscanf(file_contents, "%d.%d.%d\n", &ver, &rel, &mod);
  if ((ver > 5) || ((ver == 5) && (rel > 9)) ||
      ((ver == 5) && (rel == 9) && (mod >= 15)))
    amdgpu_capabilities = std::string("CodeObjVer4");
  free(file_contents);
  fclose(fd);
  return amdgpu_capabilities;
}

std::string _aot_get_capabilities(uint16_t vid) {
  switch (vid) {
  case 0x1002:
    return (_aot_get_amdgpu_capabilities());
    break;
  case 0x1dba:
    // FIXME return version of cuda here
    return nullptr;
    break;
  }
  return nullptr;
}

std::string _aot_get_triple(uint16_t VendorID, uint16_t DeviceID) {
  switch (VendorID) {
  case 0x1002:
    return (std::string("amdgcn-amd-amdhsa"));
    break;
  case 0x1dba:
    return (std::string("nvptx64-nvidia-cuda"));
    break;
  }
  return nullptr;
}

int main(int argc, char **argv) {
  bool print_codename = false;
  bool print_numeric = false;
  bool print_capabilities_for_runtime_requirements = false;
  bool print_long_codename = false;
  bool amdgpu_arch = false;
  bool nvidia_arch = false;
  AOT_get_all_active_devices = false;
  bool print_triple = false;
  std::string lookup_value;

  std::string a;
  for (int argi = 0; argi < argc; argi++) {
    a = std::string(argv[argi]);
    if (argi == 0) {
      // look for arch-specific invocation with symlink
      amdgpu_arch = (a.find("amdgpu-arch") != -1);
      nvidia_arch = (a.find("nvidia-arch") != -1);
    } else {
      if (a == "-n") {
        print_numeric = true;
      } else if (a == "-c") {
        print_codename = true;
      } else if (a == "-l") {
        print_long_codename = true;
      } else if (a == "-r") {
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
      } else {
        lookup_value = a;
      }
    }
  }

  std::vector<std::string> PCI_IDS;

  if (lookup_value.empty()) {
    // No lookup_value so get the current pci ids.
    // First check if invocation was arch specific.
    if (print_capabilities_for_runtime_requirements) {
      fprintf(stderr, "Error: cannot lookup offload-arch/codenane AND query\n");
      fprintf(stderr, "       active capabilities (-r).\n");
      return 1;
    }
    if (amdgpu_arch) {
      PCI_IDS = _aot_get_pci_ids("DRIVER=amdgpu");
    } else if (nvidia_arch) {
      PCI_IDS = _aot_get_pci_ids("DRIVER=nvidia");
    } else {
      // Search for all supported offload archs;
      PCI_IDS = _aot_get_pci_ids("DRIVER=amdgpu");
      if (AOT_get_all_active_devices) {
        std::vector<std::string> PCI_IDs_next_arch;
        PCI_IDs_next_arch = _aot_get_pci_ids("DRIVER=nvidia");
        for (auto PCI_ID : PCI_IDs_next_arch)
          PCI_IDS.push_back(PCI_ID);
      } else {
        // stop offload-arch at first device found`
        if (PCI_IDS.empty())
          PCI_IDS = _aot_get_pci_ids("DRIVER=nvidia");
      }
    }
  } else {
    PCI_IDS = _aot_lookup_offload_arch(lookup_value);
    if (PCI_IDS.empty())
      PCI_IDS = _aot_lookup_codename(lookup_value);
    if (PCI_IDS.empty()) {
      fprintf(stderr, "Error: Could not find \"%s\" in offload-arch tables\n",
              lookup_value.c_str());
      fprintf(stderr, "       as either an offload-arch or codename\n");
      return 1;
    }
  }

  if (PCI_IDS.empty()) {
    return 1;
  }

  std::vector<uint16_t> VendorIDs;
  std::vector<uint16_t> DeviceIDs;
  // Convert string to two uint16_t hex values
  for (auto PCI_ID : PCI_IDS) {
    unsigned vid, devid;
    sscanf(PCI_ID.c_str(), "%x:%x", &vid, &devid);
    VendorIDs.push_back(vid);
    DeviceIDs.push_back(devid);
  }

  struct pci_ids pacc;
  char namebuf[MAXPATHSIZE];
  if (print_long_codename)
    pacc = pci_ids_create();

  int i = 0;
  int rc = 0;
  for (auto vid : VendorIDs) {
    std::string target = _aot_get_target(vid, DeviceIDs[i]);
    if (target.empty()) {
      fprintf(stderr, "Could not find targetid for %x:%x\n", vid, DeviceIDs[i]);
      rc = 1;
    } else {
      std::string xinfo;
      if (print_codename)
        xinfo.append(" ").append(_aot_get_codename(vid, DeviceIDs[i]));
      if (print_numeric)
        xinfo.append(" ").append(PCI_IDS[i]);
      if (print_triple)
        xinfo.append(" ").append(_aot_get_triple(vid, DeviceIDs[i]));
      if (print_capabilities_for_runtime_requirements)
        xinfo.append(" ").append(_aot_get_capabilities(vid));
      if (print_long_codename) {
        // Use Jonathan's parsing code to find long name from pci.ids
        std::string long_code_name(
            pci_ids_lookup(pacc, namebuf, sizeof(namebuf), vid, DeviceIDs[i]));
        xinfo.append(" : ").append(long_code_name);
      }
      printf("%s%s\n", target.c_str(), xinfo.c_str());
    }
    i++;
  }

  if (print_long_codename)
    pci_ids_destroy(pacc);
  return rc;
}
