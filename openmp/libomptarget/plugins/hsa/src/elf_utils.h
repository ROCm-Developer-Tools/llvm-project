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

#pragma once

#include "hsa_ext_finalize.h"
#include <gelf.h>

typedef enum {
  STATUS_SUCCESS = 0,
  STATUS_KERNEL_INVALID_SECTION_HEADER = 1,
  STATUS_KERNEL_ELF_INITIALIZATION_FAILED = 2,
  STATUS_KERNEL_INVALID_ELF_CONTAINER = 3,
  STATUS_KERNEL_MISSING_DATA_SECTION = 4,
  STATUS_KERNEL_MISSING_CODE_SECTION = 5,
  STATUS_KERNEL_MISSING_OPERAND_SECTION = 6,
  STATUS_UNKNOWN = 7,
} status_t;

enum ElfType {
  ELF32,
  ELF64,
};

typedef enum ElfType ElfType;

struct SectionDesc {
  int sectionId;
  const char *brigName;
  const char *bifName;
};

typedef struct SectionDesc SectionDesc;

// sectionID used in brigSectionDescs
enum {
  SECTION_DATA = 0,
  SECTION_CODE,
  SECTION_OPERAND,
};

// sectionID used in hsaSectionDescs
enum {
  SECTION_BRIG = 0,
  SECTION_METADATA,
};

// typedef enum status_t status_t;

#define MODULE_t hsa_ext_module_t

#ifdef __cplusplus
extern "C" {
#endif

const SectionDesc *get_brig_section_desc(int sectionId);
const SectionDesc *get_hsa_section_desc(int sectionId);

status_t extract_section_copy(Elf *elfP, Elf_Data *secHdr,
                              const SectionDesc *desc, void **section_ptr,
                              size_t *size_ptr);

status_t create_brig_module_from_brig_file(const char *file_name,
                                           MODULE_t **brig_module);
status_t create_brig_module_from_brig_memory(char *buffer,
                                             const size_t buffer_size,
                                             MODULE_t **brig_module);

void destroy_brig_module(MODULE_t *brig_module);

#ifdef __cplusplus
}
#endif
