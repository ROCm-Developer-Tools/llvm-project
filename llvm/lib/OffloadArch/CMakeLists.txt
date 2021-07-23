


# File : /home/grodgers/git/aomp13/llvm-project/llvm/lib/OffloadArch/CMakeLists.txt

cmake_policy(SET CMP0002 NEW)
cmake_minimum_required(VERSION 3.5)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated_offload_arch.h
   COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/make_generated_offload_arch_h.sh ${CMAKE_CURRENT_SOURCE_DIR}
   DEPENDS 
     ${CMAKE_CURRENT_SOURCE_DIR}/make_generated_offload_arch_h.sh
     ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu/pciid2codename.txt 
     ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu/codename2offloadarch.txt 
     ${CMAKE_CURRENT_SOURCE_DIR}/nvidia/pciid2codename.txt 
     ${CMAKE_CURRENT_SOURCE_DIR}/nvidia/codename2offloadarch.txt 
     ${CMAKE_CURRENT_SOURCE_DIR}/intelhd/pciid2codename.txt 
     ${CMAKE_CURRENT_SOURCE_DIR}/intelhd/codename2offloadarch.txt 
)

add_custom_target(
  generated-table
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_offload_arch.h)

set(AOT_sources 
  ${CMAKE_CURRENT_SOURCE_DIR}/OffloadArch.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu/vendor_specific_capabilities.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nvidia/vendor_specific_capabilities.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/intelhd/vendor_specific_capabilities.cpp
)

add_llvm_component_library(LLVMOffloadArch
  STATIC
  ${AOT_sources}
  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/OffloadArch"
  DEPENDS generated-table
)
target_link_libraries(LLVMOffloadArch LLVMObject LLVMSupport )

include_directories(${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu
  ${CMAKE_CURRENT_SOURCE_DIR}/nvidia
  ${CMAKE_CURRENT_SOURCE_DIR}/intelhd)

add_subdirectory(offload-arch)

