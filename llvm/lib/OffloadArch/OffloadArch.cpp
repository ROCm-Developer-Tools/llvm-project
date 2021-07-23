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
//
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/WithColor.h"
#include <dirent.h>
#include <fstream>
#include "generated_offload_arch.h"
#include "llvm/OffloadArch/OffloadArch.h"

using namespace llvm;
using namespace object;

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

LLVM_ATTRIBUTE_NORETURN inline void
exitWithError(const Twine &Message, StringRef Whence = StringRef(),
              StringRef Hint = StringRef()) {
  WithColor::error(errs(), "offload-arch");
  if (!Whence.empty())
    errs() << Whence.str() << ": ";
  errs() << Message << "\n";
  if (!Hint.empty())
    WithColor::note() << Hint.str() << "\n";
  ::exit(EXIT_FAILURE);
}
LLVM_ATTRIBUTE_NORETURN inline void
exitWithError(std::error_code EC, StringRef Whence = StringRef()) {
  exitWithError(EC.message(), Whence);
}
LLVM_ATTRIBUTE_NORETURN inline void exitWithError(Error E, StringRef Whence) {
  exitWithError(errorToErrorCode(std::move(E)), Whence);
}
template <typename T, typename... Ts>
T unwrapOrError(Expected<T> EO, Ts &&...Args) {
  if (EO)
    return std::move(*EO);
  exitWithError(EO.takeError(), std::forward<Ts>(Args)...);
}

/// Function used by offload-arch tool to get requirements from each image of
/// an elf binary file. Requirements (like offload arch name, target features)
/// are read from a custom section ".offload_arch_list" in elf binary.
std::vector<std::string>
_aot_get_requirements_from_file(const std::string &input_filename) {
  std::vector<std::string> results;
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrError =
      MemoryBuffer::getFile(input_filename);
  if (!BufOrError) {
    fprintf(stderr, " MemoryBuffer error reading file \n");
    results.push_back("MEM ERROR");
    return results;
  }
  std::unique_ptr<MemoryBuffer> FileReadBuffer = std::move(*BufOrError);
  Expected<std::unique_ptr<Binary>> BinaryOrErr =
      createBinary(FileReadBuffer->getMemBufferRef(), /*Context=*/nullptr,
                   /*InitContent=*/false);
  if (!BinaryOrErr) {
    results.push_back("createBinary ERROR");
    return results;
  }
  std::unique_ptr<Binary> Bin = std::move(*BinaryOrErr);
  if (!isa<ELFObjectFile<ELF64LE>>(Bin)) {
    results.push_back("NOT ELF64LE");
    return results;
  }
  ELFObjectFile<ELF64LE> *elf_obj_file =
      dyn_cast<ELFObjectFile<ELF64LE>>(Bin.get());
  StringRef FileName = elf_obj_file->getFileName();
  for (section_iterator SI = elf_obj_file->section_begin(),
                        SE = elf_obj_file->section_end();
       SI != SE; ++SI) {
    const SectionRef &Section = *SI;
    StringRef SectionName = unwrapOrError(Section.getName(), FileName);
    if (SectionName == ".offload_arch_list") {
      StringRef Contents = unwrapOrError(Section.getContents(), FileName);
      const char *arch_list_ptr = Contents.data();
      std::string arch;
      // Iterate over list of requirements to extract individual requirements.
      for (uint i = 0; i < Contents.size(); i++) {
        for (uint j = i; arch_list_ptr[j] != '\0'; j++, i++) {
          arch.push_back(arch_list_ptr[i]);
        }
        results.push_back(arch);
        arch.resize(0);
      }
    }
  }
  return results;
}
