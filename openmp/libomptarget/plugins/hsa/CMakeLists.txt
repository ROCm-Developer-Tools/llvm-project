##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for an HSA machine if available.
#
##===----------------------------------------------------------------------===##

################################################################################
# Add check for required compiler

find_package(LLVM QUIET CONFIG
  PATHS
  $ENV{AOMP}
  $ENV{HOME}/rocm/aomp
  /opt/rocm/aomp
  /usr/lib/rocm/aomp
  ${LIBOMPTARGET_NVPTX_CUDA_COMPILER_DIR}
  ${LIBOMPTARGET_NVPTX_CUDA_LINKER_DIR}
  ${CMAKE_CXX_COMPILER_DIR}
  NO_DEFAULT_PATH
  )

if (LLVM_DIR)
  libomptarget_say("Found LLVM ${LLVM_PACKAGE_VERSION}. Configure: ${LLVM_DIR}/LLVMConfig.cmake")
else()
  libomptarget_say("Not building HSA plugin: AOMP llvm not found")
  return()
endif()

set(AOMP_DIR_FOUND ${LLVM_DIR})
set(AOMP_INSTALL_PREFIX ${LLVM_INSTALL_PREFIX})
set(AOMP_MAIN_INCDIR ${LLVM_BUILD_MAIN_INCLUDE_DIR})
set(AOMP_BINDIR ${AOMP_INSTALL_PREFIX}/bin)
set(AOMP_INCDIR ${AOMP_INSTALL_PREFIX}/include)
set(AOMP_LIBDIR ${AOMP_INSTALL_PREFIX}/lib)

find_package(Clang QUIET CONFIG
  PATHS
  $ENV{AOMP}
  $ENV{HOME}/rocm/aomp
  /opt/rocm/aomp
  /usr/lib/rocm/aomp
  ${LIBOMPTARGET_NVPTX_CUDA_COMPILER_DIR}
  ${LIBOMPTARGET_NVPTX_CUDA_LINKER_DIR}
  ${CMAKE_CXX_COMPILER_DIR}
  NO_DEFAULT_PATH
  )

if (CLANG_CMAKE_DIR)
  libomptarget_say("Found Clang ${LLVM_PACKAGE_VERSION}. Configure: ${CLANG_CMAKE_DIR}/ClangConfig.cmake")
else()
  libomptarget_say("Not building HSA plugin: AOMP clang not found")
  return()
endif()

# Add check for required libraries

if(NOT LIBOMPTARGET_DEP_LIBELF_FOUND)
  libomptarget_say("Not building HSA plugin: LIBELF not found")
  return()
endif()

################################################################################
# Looking for ROCM...
################################################################################
pkg_check_modules(LIBOMPTARGET_SEARCH_LIBHSA QUIET libhsa-runtime64)

set(LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS $ENV{HSA_RUNTIME_PATH}/include/hsa)

# This is usually found on /opt/rocm/lib or /usr/lib/rocm/lib
find_path (
  LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS
  NAMES
  libhsa-runtime64.so
  PATHS
  $ENV{HSA_RUNTIME_PATH}/lib
  $ENV{HOME}/rocm/aomp/lib
  /opt/rocm/aomp/lib
  /opt/rocm/lib
  /usr/lib/rocm/lib
  )

find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBHSA
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS
  LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS
  LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS)

if(NOT LIBOMPTARGET_DEP_LIBHSA_FOUND)
  libomptarget_say("Not building HSA plugin: LIBHSA not found")
  return()
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  libomptarget_say("Not building HSA plugin: only support HSA in Linux x86_64, ppc64le , or aarch64 hosts.")
  return()
endif()
libomptarget_say("Building HSA offloading plugin")

################################################################################
# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=HSA)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(ppc64le)|(aarch64)$")
   add_definitions(-DLITTLEENDIAN_CPU=1)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions(-DHSA_ERROR_REPORT)
endif()

include_directories(
  ${LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
)

add_library(omptarget.rtl.hsa SHARED
  src/rtl.cpp 
  src/elf_utils.c )

# Install plugin under the lib destination folder.
# When we build for debug, OPENMP_LIBDIR_SUFFIX get set to -debug
install(TARGETS omptarget.rtl.hsa LIBRARY DESTINATION "lib${OPENMP_LIBDIR_SUFFIX}")

# We need the AOMP specific build of ATMI that has HSA_INTEROP turned on. 
# Also, the AOMP specific build of ATMI has seperate release and debug builds. 
target_link_libraries(
  omptarget.rtl.hsa
  -L${AOMP_LIBDIR}${OPENMP_LIBDIR_SUFFIX} -latmi_runtime -Wl,-rpath,${AOMP_LIBDIR}${OPENMP_LIBDIR_SUFFIX}
  -L${LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS} -lhsa-runtime64 -lhsakmt  -Wl,-rpath,${LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS} 
  -L${AOMP_LIBDIR} -lLLVMAMDGPUDesc -lLLVMAMDGPUUtils -lLLVMMC -lLLVMCore -lLLVMSupport -Wl,-rpath,${AOMP_LIBDIR}
  -lelf
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
  )

# Report to the parent scope that we are building a plugin for hsa
set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} amdgcn-amd-amdhsa" PARENT_SCOPE)

