#ifndef __OFFLOAD_ARCH_H__
#define __OFFLOAD_ARCH_H__
#include "generated_offload_arch.h"
#include <iostream>
#include <string>
#include <vector>
std::string _aot_get_file_contents(std::string fname);
std::string _aot_amdgpu_capabilities(uint16_t vid, uint16_t devid,
                                     std::string oa);
std::string _aot_nvidia_capabilities(uint16_t vid, uint16_t devid,
                                     std::string oa);
std::vector<std::string>
_aot_get_requirements_from_file(const std::string &filename);

#endif
