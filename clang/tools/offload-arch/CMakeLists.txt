
project(offload-arch)

cmake_policy(SET CMP0002 NEW)
cmake_minimum_required(VERSION 3.5)

add_executable(offload-arch OffloadArch.cpp)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

# needed to find the generated header.
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated_offload_arch.h
   COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/make_generated_offload_arch_h.sh ${CMAKE_CURRENT_SOURCE_DIR}
   DEPENDS 
     ${CMAKE_CURRENT_SOURCE_DIR}/make_generated_offload_arch_h.sh
     ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu/pciid2codename.txt 
     ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu/codename2offloadarch.txt 
     ${CMAKE_CURRENT_SOURCE_DIR}/nvidia/pciid2codename.txt 
     ${CMAKE_CURRENT_SOURCE_DIR}/nvidia/codename2offloadarch.txt 
)
add_custom_target(
  generated-table
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_offload_arch.h)

add_custom_target(
   symlink1 ALL
   COMMAND ln -sf offload-arch amdgpu-arch
   DEPENDS offload-arch)
add_custom_target(
   symlink2 ALL
   COMMAND ln -sf offload-arch nvidia-arch
   DEPENDS offload-arch)

add_dependencies(offload-arch generated-table)

install(PROGRAMS   
    ${CMAKE_CURRENT_BINARY_DIR}/amdgpu-arch
    ${CMAKE_CURRENT_BINARY_DIR}/nvidia-arch
     DESTINATION bin)

