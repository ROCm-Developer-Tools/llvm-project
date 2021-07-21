


# File : /home/grodgers/git/aomp13/llvm-project/llvm/lib/OffloadArch/CMakeLists.txt

cmake_policy(SET CMP0002 NEW)
cmake_minimum_required(VERSION 3.5)

set(AOT_sources 
  ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OffloadArch.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu/vendor_specific_capabilities.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nvidia/vendor_specific_capabilities.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/intelhd/vendor_specific_capabilities.cpp
)

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

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  # For standalone in build subdirectory, not part of llvm build
  project(offload-arch)
  add_executable(offload-arch 
    ${AOT_sources}
  )
  add_dependencies(offload-arch generated-table)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)
else()
  # This is part of llvm build
  add_llvm_component_library(LLVMOffloadArch
    ${AOT_sources}
    ADDITIONAL_HEADER_DIRS
    "${LLVM_MAIN_INCLUDE_DIR}/llvm/OffloadArch"
    DEPENDS generated-table
  )
  add_llvm_tool(offload-arch 
    ${AOT_sources}
    DEPENDS generated-table
  )
endif()

target_link_libraries(offload-arch PUBLIC -ldl -lelf)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

# need this include to find the generated header.
include_directories(${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu
  ${CMAKE_CURRENT_SOURCE_DIR}/nvidia
  ${CMAKE_CURRENT_SOURCE_DIR}/intelhd)

add_custom_target(
   amdgpu-arch ALL
   COMMAND ln -sf offload-arch amdgpu-arch
   DEPENDS offload-arch)
add_custom_target(
   nvidia-arch ALL
   COMMAND ln -sf offload-arch nvidia-arch
   DEPENDS offload-arch)
add_custom_target(
   intelhd-arch ALL
   COMMAND ln -sf offload-arch intelhd-arch
   DEPENDS offload-arch)


install(PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/amdgpu-arch
    ${CMAKE_CURRENT_BINARY_DIR}/nvidia-arch
    ${CMAKE_CURRENT_BINARY_DIR}/intelhd-arch
     DESTINATION bin
     COMPONENT offload-arch)

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  install(PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/offload-arch
     DESTINATION bin
     COMPONENT offload-arch)
endif()
