//===-- offload-arch/getRequirementsFromFile.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file clang/tools/offload-arch/getRequirementsFromFile.cpp
/// Function used by offload-arch tool to get requirements from each image of
/// an elf binary file.
///
//===----------------------------------------------------------------------===//

#include "OffloadArch.h"
#include <fcntl.h>
#include <gelf.h>
#include <libelf.h>

std::vector<std::string>
_aot_get_requirements_from_file(std::string input_filename) {
  std::vector<std::string> results;
  int ii, num_syms, fd;
  elf_version(EV_CURRENT);
  fd = open(input_filename.c_str(), O_RDONLY);
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

  // Find symbol talble section
  GElf_Shdr symtab;
  Elf_Scn *scn = NULL;
  while ((scn = elf_nextscn(elf, scn)) != NULL) {
    gelf_getshdr(scn, &symtab);
    if (symtab.sh_type == SHT_SYMTAB)
      break;
  }
  if (!scn) {
    fprintf(stderr, "ERROR: No symbol table found in %s\n",
            input_filename.c_str());
    return results;
  }

  // Get file offsets for requirements string of each offload image in
  // application binary by looking for "offload_arch" in symtab
  Elf_Data *sec_data = elf_getdata(scn, NULL);
  num_syms = symtab.sh_size / symtab.sh_entsize;
  int num_images = 0;
  std::string symname;
  GElf_Sym sym;
  size_t req_offset, req_strsz;
  std::vector<size_t> req_offsets, req_strszs;
  for (ii = 0; ii < num_syms; ++ii) {
    gelf_getsym(sec_data, ii, &sym);
    symname = std::string(elf_strptr(elf, symtab.sh_link, sym.st_name));
    std::size_t found_loc = symname.find("offload_arch");
    if (found_loc != std::string::npos) {
      GElf_Shdr oshdr, *osh;
      Elf_Scn *oscn;
      oscn = elf_getscn(elf, sym.st_shndx);
      osh = gelf_getshdr(oscn, &oshdr);
      req_offset = (size_t)osh->sh_offset + (sym.st_value - osh->sh_addr);
      req_strsz = sym.st_size;
      req_offsets.push_back(req_offset);
      req_strszs.push_back(req_strsz);
      num_images++;
    }
  }
  elf_end(elf);

  if (!num_images) {
    fprintf(stderr, "ERROR: No requirements found for any images in %s\n",
            input_filename.c_str());
    return results;
  }

  // Reopen file and seek to the requirements strings
  FILE *pFile;
  pFile = fopen(input_filename.c_str(), "rb");
  for (ii = 0; ii < num_images; ++ii) {
    req_offset = req_offsets[ii];
    req_strsz = req_strszs[ii];
    fseek(pFile, req_offset, SEEK_SET);
    char buffer[1024];
    size_t fread_result = fread(buffer, 1, req_strsz, pFile);
    buffer[(uint)req_strsz] = '\0';
    results.push_back(std::string(buffer));
  }

  fclose(pFile);
  return results;
}
