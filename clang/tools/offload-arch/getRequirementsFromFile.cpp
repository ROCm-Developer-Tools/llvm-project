//===------- offload-arch/getRequirementsFromFile.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
///
/// \file clang/tools/offload-arch/getRequirementsFromFile.cpp
/// Function used by offload-arch tool to get requirements from each image of
/// an elf binary file. Requirements (like offload arch name, target features)
/// are read from a custom section ".offload_arch_list" in elf binary.
//===---------------------------------------------------------------------===//

#include "OffloadArch.h"
#include <fcntl.h>
#include <gelf.h>
#include <libelf.h>
#include <unistd.h>

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
    if(gelf_getshdr(scn, &offload_section) != &offload_section) {
      fprintf(stderr, "ERROR: No section header found in %s\n",
            input_filename.c_str());
      return results;
    }

    // Read section name from pointer to section header.
    std::string sname(elf_strptr(elf, section_idx, offload_section.sh_name));
    if(sname.empty()) {
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
  if(!sec_data) {
    fprintf(stderr, "ERROR: list of offload archs not found in %s\n",
            input_filename.c_str());
    return results;
  }
  char *arch_list_ptr = (char *)sec_data->d_buf;

  // Iterate over list of requirements to extract individual requirements.
  for(uint i = 0; i < offload_section.sh_size; i++) {
    for(uint j = i; arch_list_ptr[j] != '\0'; j++, i++) {
      arch.push_back(arch_list_ptr[i]);
    }
    results.push_back(arch);
    arch.resize(0);
  }
  elf_end(elf);
  close(fd);
  return results;
}
