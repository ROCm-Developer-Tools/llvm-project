//===-- llvm/lib/OffloadArch/OffloadArch.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// OffloadArch.cpp : Library Functions for OffloadArch
///
//===----------------------------------------------------------------------===//

#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <gelf.h>
#include <libelf.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "generated_offload_arch.h"
#include "llvm/OffloadArch/OffloadArch.h"

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
  for (const AOT_CODENAME_ID_TO_STRING id2str : AOT_CODENAMES)
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
  case 0x8086:
    capabilities.append(_aot_intelhd_capabilities(vid, devid, oa));
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
  case 0x8086:
    return (std::string("spir64-intel-unknown"));
    break;
  }
  return retval;
}

std::vector<std::string> _aot_get_all_pci_ids() {
  std::vector<std::string> PCI_IDS =
      _aot_get_pci_ids(AMDGPU_SEARCH_PHRASE, AMDGPU_PCIID_PHRASE);
  for (auto PCI_ID :
       _aot_get_pci_ids(NVIDIA_SEARCH_PHRASE, NVIDIA_PCIID_PHRASE))
    PCI_IDS.push_back(PCI_ID);
  for (auto PCI_ID :
       _aot_get_pci_ids(INTELHD_SEARCH_PHRASE, INTELHD_PCIID_PHRASE))
    PCI_IDS.push_back(PCI_ID);
  return PCI_IDS;
}

/// Get runtime capabilities of this system for libomptarget runtime
extern "C" int
__aot_get_capabilities_for_runtime(char *offload_arch_output_buffer,
                                   size_t offload_arch_output_buffer_size) {
  std::vector<std::string> PCI_IDS = _aot_get_all_pci_ids();
  std::string offload_arch;
  for (auto PCI_ID : PCI_IDS) {
    unsigned vid32, devid32;
    sscanf(PCI_ID.c_str(), "%x:%x", &vid32, &devid32);
    uint16_t vid = vid32;
    uint16_t devid = devid32;
    offload_arch = _aot_get_offload_arch(vid, devid);
    if (offload_arch.empty()) {
      fprintf(stderr, "ERROR: offload-arch not found for %x:%x.\n", vid, devid);
      return 1;
    }
    std::string caps = _aot_get_capabilities(vid, devid, offload_arch);
    std::size_t found_loc = caps.find("NOT-VISIBLE");
    if (found_loc == std::string::npos) {
      // Found first visible GPU, so append caps and exit loop
      offload_arch.append(caps);
      break;
    }
  }
  size_t out_str_len = strlen(offload_arch.c_str());
  if (out_str_len > offload_arch_output_buffer_size) {
    fprintf(stderr, "ERROR: strlen %ld exceeds buffer length %ld \n",
            out_str_len, offload_arch_output_buffer_size);
    return 1;
  }
  strncpy(offload_arch_output_buffer, offload_arch.c_str(), out_str_len);
  offload_arch_output_buffer[out_str_len] = '\0'; // terminate string
  return 0;
}

/// Function used by offload-arch tool to get requirements from each image of
/// an elf binary file. Requirements (like offload arch name, target features)
/// are read from a custom section ".offload_arch_list" in elf binary.

std::vector<std::string>
_aot_get_requirements_from_file(const std::string &input_filename) {
  std::vector<std::string> results;
  int fd;
  elf_version(EV_CURRENT);
  fd = open(input_filename.c_str(), O_RDONLY | O_LARGEFILE);
  if (!fd) {
    fprintf(stderr, "ERROR: Could not open file %s\n", input_filename.c_str());
    return results;
  }
  Elf *elf = elf_begin(fd, ELF_C_READ, NULL);
  if (!elf) {
    fprintf(stderr, "ERROR: File %s is NOT a valid elf file\n",
            input_filename.c_str());
    return results;
  }

  GElf_Shdr offload_section;
  Elf_Scn *scn = NULL;
  size_t section_idx;
  uint status;
  // Get section header index.
  status = elf_getshdrstrndx(elf, &section_idx);
  if (status) {
    fprintf(stderr, "ERROR: No section header index found in %s\n",
            input_filename.c_str());
    return results;
  }

  // Iterate over sections of elf.
  while ((scn = elf_nextscn(elf, scn)) != NULL) {

    // Get pointer to section header.
    if (gelf_getshdr(scn, &offload_section) != &offload_section) {
      fprintf(stderr, "ERROR: No section header found in %s\n",
              input_filename.c_str());
      return results;
    }

    // Read section name from pointer to section header.
    std::string sname(elf_strptr(elf, section_idx, offload_section.sh_name));
    if (sname.empty()) {
      fprintf(stderr, "ERROR: No section name found in %s\n",
              input_filename.c_str());
      return results;
    }

    // Check if this is the section being searched.
    if (!sname.compare(".offload_arch_list"))
      break;
  }
  if (!scn) {
    fprintf(stderr, "ERROR: .offload_arch_list section not found in %s\n",
            input_filename.c_str());
    return results;
  }

  Elf_Data *sec_data = NULL;
  std::string arch;

  // Get a pointer to list of null terminated requirements.
  sec_data = elf_getdata(scn, sec_data);
  if (!sec_data) {
    fprintf(stderr, "ERROR: list of offload archs not found in %s\n",
            input_filename.c_str());
    return results;
  }
  char *arch_list_ptr = (char *)sec_data->d_buf;

  // Iterate over list of requirements to extract individual requirements.
  for (uint i = 0; i < offload_section.sh_size; i++) {
    for (uint j = i; arch_list_ptr[j] != '\0'; j++, i++) {
      arch.push_back(arch_list_ptr[i]);
    }
    results.push_back(arch);
    arch.resize(0);
  }
  elf_end(elf);
  close(fd);
  return results;
}
