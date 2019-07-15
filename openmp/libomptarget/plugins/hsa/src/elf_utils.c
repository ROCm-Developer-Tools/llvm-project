/* Copyright 2014 HSA Foundation Inc.  All Rights Reserved.
 *
 * HSAF is granting you permission to use this software and documentation (if
 * any) (collectively, the "Materials") pursuant to the terms and conditions
 * of the Software License Agreement included with the Materials.  If you do
 * not have a copy of the Software License Agreement, contact the  HSA
 * Foundation for a copy. Redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * WITH THE SOFTWARE.
 */

//===----------------------------------------------------------------------===//
//
// Utility functions, most of them are from HSA examples
// github: guansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//

#include <libelf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hsa.h"
#include "hsa_ext_finalize.h"

#include "elf_utils.h"

MODULE_t __hsaBrigModule;

SectionDesc brigSectionDescs[] = {
    {SECTION_DATA, "hsa_data", ".brig_hsa_data"},
    {SECTION_CODE, "hsa_code", ".brig_hsa_code"},
    {SECTION_OPERAND, "hsa_operand", ".brig_hsa_operand"},
};

SectionDesc hsaSectionDescs[] = {
    {SECTION_BRIG, ".brig", ".brig_hsa_data"},
    {SECTION_METADATA, ".AMDGPU.runtime_metadata", ".AMDGPU.runtime_metadata"},
};

extern int fileno(FILE *stream);

const SectionDesc *get_brig_section_desc(int sectionId) {
  const int NUM_PREDEFINED_SECTIONS =
      sizeof(brigSectionDescs) / sizeof(brigSectionDescs[0]);
  for (int i = 0; i < NUM_PREDEFINED_SECTIONS; ++i) {
    if (brigSectionDescs[i].sectionId == sectionId) {
      return &brigSectionDescs[i];
    }
  }
  return NULL;
}

const SectionDesc *get_hsa_section_desc(int sectionId) {
  const int NUM_PREDEFINED_SECTIONS =
      sizeof(hsaSectionDescs) / sizeof(hsaSectionDescs[0]);
  for (int i = 0; i < NUM_PREDEFINED_SECTIONS; ++i) {
    if (hsaSectionDescs[i].sectionId == sectionId) {
      return &hsaSectionDescs[i];
    }
  }
  return NULL;
}

static Elf_Scn *get_elf_section(Elf *elfP, Elf_Data *secHdr,
                                const SectionDesc *desc) {

  int cnt = 0;
  Elf_Scn *scn = NULL;

  Elf64_Shdr *shdr64 = NULL;
  Elf32_Shdr *shdr32 = NULL;

  ElfType type;

  Elf64_Ehdr *eh64 = elf64_getehdr(elfP);
  Elf32_Ehdr *eh32 = elf32_getehdr(elfP);
  if (eh64 && !eh32) {
    type = ELF64;
  } else if (eh32 && !eh64) {
    type = ELF32;
  } else {
    printf("Ambiguous ELF header!\n");
  }

  char *sectionName = NULL;

  /* Iterate thru the elf sections */
  for (cnt = 1, scn = NULL; (scn = elf_nextscn(elfP, scn)); cnt++) {
    if (type == ELF64) {
      if ((shdr64 = elf64_getshdr(scn)) == NULL) {
        return NULL;
      }
      sectionName = (char *)secHdr->d_buf + shdr64->sh_name;
    } else if (type == ELF32) {
      if ((shdr32 = elf32_getshdr(scn)) == NULL) {
        return NULL;
      }
      sectionName = (char *)secHdr->d_buf + shdr32->sh_name;
    } else {
      printf("Ambiguous ELF header!\n");
    }

    if (sectionName && ((strcmp(sectionName, desc->brigName) == 0) ||
                        (strcmp(sectionName, desc->bifName) == 0))) {
      // printf("Found Elf section name: %s\n", sectionName);
      return scn;
    } else {
      // printf("Elf section name: %s\n", sectionName);
    }
  }

  return NULL;
}

/* Extract section and copy into HsaBrig */
status_t extract_section_copy(Elf *elfP, Elf_Data *secHdr,
                              const SectionDesc *desc, void **section_ptr,
                              size_t *size_ptr) {

  Elf_Scn *scn = NULL;
  Elf_Data *data = NULL;

  void *address_to_copy;
  size_t section_size = 0;

  scn = get_elf_section(elfP, secHdr, desc);

  if (scn) {
    if ((data = elf_getdata(scn, NULL)) == NULL) {
      return STATUS_UNKNOWN;
    }

    section_size = data->d_size;
    // printf("Found Elf section size: %lu\n", section_size);

    if (section_size > 0) {
      address_to_copy = malloc(section_size);
      memcpy(address_to_copy, data->d_buf, section_size);
    }
  }

  if ((!scn || section_size == 0)) {
    // printf("Not Found Elf section size: %lu\n", section_size);
    return STATUS_UNKNOWN;
  }

  *section_ptr = address_to_copy;
  *size_ptr = section_size;

  // Crazy pointer casting!
  // __hsaBrigModule=(MODULE_t)address_to_copy;

  return STATUS_SUCCESS;
}

/* Reads binary of BRIG and BIF format */
static status_t read_binary_elf(MODULE_t **brig_module_address_ptr, Elf *elfP) {

  status_t status;

  ElfType type;

  Elf_Data *secHdr = NULL;
  Elf_Scn *scn = NULL;

  if (elf_kind(elfP) != ELF_K_ELF) {
    return STATUS_KERNEL_INVALID_ELF_CONTAINER;
  }

  /*Need MachineID again, clean this when we have a clear view on the backend
   * format */
  uint16_t MachineID;
  {
    Elf64_Ehdr *eh64 = elf64_getehdr(elfP);
    Elf32_Ehdr *eh32 = elf32_getehdr(elfP);
    if (eh64 && !eh32) {
      MachineID = eh64->e_machine;
      type = ELF64;
    } else if (eh32 && !eh64) {
      MachineID = eh32->e_machine;
      type = ELF32;
    } else {
      printf("Ambiguous ELF header!\n");
      return STATUS_KERNEL_INVALID_ELF_CONTAINER;
    }
  }

  switch (MachineID) {
  case 0: {
    return STATUS_KERNEL_MISSING_CODE_SECTION;
  } break;
  case 44890: {
    size_t section_size;
    Elf32_Ehdr *ehdr = NULL;

    /* Obtain the .shstrtab data buffer */
    if (((ehdr = elf32_getehdr(elfP)) == NULL) ||
        ((scn = elf_getscn(elfP, ehdr->e_shstrndx)) == NULL) ||
        ((secHdr = elf_getdata(scn, NULL)) == NULL)) {
      return STATUS_KERNEL_INVALID_SECTION_HEADER;
    }

    status =
        extract_section_copy(elfP, secHdr, get_hsa_section_desc(SECTION_BRIG),
                             (void **)brig_module_address_ptr, &section_size);

    if (status != STATUS_SUCCESS) {
      return STATUS_KERNEL_MISSING_DATA_SECTION;
    }

    elf_end(elfP);

    return STATUS_SUCCESS;
  } break;

  case 44891: {
    size_t section_size;
    Elf64_Ehdr *ehdr = NULL;

    /* Obtain the .shstrtab data buffer */
    if (((ehdr = elf64_getehdr(elfP)) == NULL) ||
        ((scn = elf_getscn(elfP, ehdr->e_shstrndx)) == NULL) ||
        ((secHdr = elf_getdata(scn, NULL)) == NULL)) {
      return STATUS_KERNEL_INVALID_SECTION_HEADER;
    }

    status =
        extract_section_copy(elfP, secHdr, get_hsa_section_desc(SECTION_BRIG),
                             (void **)brig_module_address_ptr, &section_size);

    if (status != STATUS_SUCCESS) {
      return STATUS_KERNEL_MISSING_DATA_SECTION;
    }

    elf_end(elfP);

    return STATUS_SUCCESS;
  } break;

  default:
    printf("Unsupported Machine ID! %d\n", MachineID);
  }

  return STATUS_KERNEL_INVALID_ELF_CONTAINER;
}

/*
 * Loads a BRIG module from a specified file. This
 * function does not validate the module.
 */
static int load_module_from_file(const char *file_name,
                                 hsa_ext_module_t *module) {
  int rc = -1;

  FILE *fp = fopen(file_name, "rb");

  rc = fseek(fp, 0, SEEK_END);

  size_t file_size = (size_t)(ftell(fp) * sizeof(char));

  rc = fseek(fp, 0, SEEK_SET);

  char *buf = (char *)malloc(file_size);

  memset(buf, 0, file_size);

  size_t read_size = fread(buf, sizeof(char), file_size, fp);

  if (read_size != file_size) {
    free(buf);
  } else {
    rc = 0;
    *module = (hsa_ext_module_t)buf;
  }

  fclose(fp);

  return rc;
}

/*
 * Load the real BRIG binary from a file.
 */
// const char * filename=".so.tgt-hsail64.brig";
// load_module_from_file(filename, &__hsaBrigModule);
// DP("Load module from file: %s\n", filename);

/*
 * Create HSA brig module
 */
status_t
create_brig_module_from_brig_memory(char *buffer, const size_t buffer_size,
                                    MODULE_t **brig_module_address_ptr) {
  Elf *elfP = NULL;

  status_t status;

  if (elf_version(EV_CURRENT) == EV_NONE) {
    status = STATUS_KERNEL_ELF_INITIALIZATION_FAILED;
  } else {
    if ((elfP = elf_memory(buffer, buffer_size)) == NULL) {
      status = STATUS_KERNEL_INVALID_ELF_CONTAINER;
    } else {
      status = read_binary_elf(brig_module_address_ptr, elfP);
    }
  }

  // printf("---------------- brig addr %016llx.\n", (long long
  // unsigned)(Elf64_Addr)(brig_module)); printf("---------------- brig
  // %016llx.\n", (long long unsigned)(Elf64_Addr)(*brig_module));

  if (status != STATUS_SUCCESS) {
    printf("Could not create BRIG module: %d\n", status);
    if (status == STATUS_KERNEL_INVALID_SECTION_HEADER ||
        status == STATUS_KERNEL_ELF_INITIALIZATION_FAILED ||
        status == STATUS_KERNEL_INVALID_ELF_CONTAINER) {
      printf("The ELF file is invalid or possibley corrupted.\n");
    }
    if (status == STATUS_KERNEL_MISSING_DATA_SECTION ||
        status == STATUS_KERNEL_MISSING_CODE_SECTION ||
        status == STATUS_KERNEL_MISSING_OPERAND_SECTION) {
      printf("One or more ELF sections are missing. Use readelf command to \
          to check if hsa_data, hsa_code and hsa_operands exist.\n");
    }
  }

  return status;
}

/*
 * Destroy HSA brig module
 */
void destroy_brig_module(MODULE_t *brig_module) { free(brig_module); }
