##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# amdgcn/CMakeLists.txt
# Written by Greg Rodgers (Gregory.Rodgers@amd.com)
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
  libomptarget_say("Not building AMDGCN device RTL: AOMP not found")
  return()
endif()

set(AOMP_DIR_FOUND ${LLVM_DIR})
set(AOMP_INSTALL_PREFIX ${LLVM_INSTALL_PREFIX})
set(AOMP_MAIN_INCDIR ${LLVM_BUILD_MAIN_INCLUDE_DIR})

if (AOMP_INSTALL_PREFIX)
  set(AOMP_BINDIR ${AOMP_INSTALL_PREFIX}/bin)
  set(AOMP_INCDIR ${AOMP_INSTALL_PREFIX}/include)
  set(AOMP_LIBDIR ${AOMP_INSTALL_PREFIX}/lib)
else()
  set(AOMP_BINDIR ${LLVM_BUILD_BINARY_DIR}/bin)
  set(AOMP_INCDIR ${LLVM_BUILD_BINARY_DIR}/include)
  set(AOMP_LIBDIR ${LLVM_LIBRARY_DIRS})
endif()

# Pass check
libomptarget_say("Building AMDGCN device RTL. LLVM_COMPILER_PATH=${AOMP_BINDIR} ")

###############################################################################
# Main project
###############################################################################

project(omptarget-amdgcn)

add_custom_target(omptarget-amdgcn ALL)

#optimization level
set(optimization_level 2)

# Activate RTL message dumps if requested by the user.
if(LIBOMPTARGET_NVPTX_DEBUG)
  set(CUDA_DEBUG -DOMPTARGET_NVPTX_DEBUG=-1)
endif()

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll)

# for both in-tree and out-of-tree build
if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
else()
  set(OUTPUTDIR ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endif()

# create libraries
set(mcpus $ENV{GFXLIST})

if(mcpus)
else()
  set(mcpus gfx700 gfx701 gfx801 gfx803 gfx900)
endif()

separate_arguments(mcpus)

macro(collect_sources name dir)
  set(cuda_sources)
  set(ocl_sources)
  set(llvm_sources)
  foreach(file ${ARGN})
    file(RELATIVE_PATH rfile ${dir} ${file})
    get_filename_component(rdir ${rfile} DIRECTORY)
    get_filename_component(fname ${rfile} NAME_WE)
    get_filename_component(fext ${rfile} EXT)
    #file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${rdir})
    if (fext STREQUAL ".cu")
      set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${rdir}/${fname}.cu)
      list(APPEND cuda_sources ${cfile})
    endif()

    if (fext STREQUAL ".cl")
      set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${rdir}/${fname}.cl)
      list(APPEND ocl_sources ${cfile})
    endif()
    if (fext STREQUAL ".ll")
      list(APPEND csources ${file})
      set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${rdir}/${fname}.ll)
      list(APPEND llvm_sources ${cfile})
    endif()
  endforeach()
endmacro()

macro(add_cuda_bc_library name dir)
  set(cu_cmd ${AOMP_BINDIR}/clang++
    -std=c++11
    -fcuda-rdc
    -fvisibility=default
    --cuda-device-only
    -Wno-unused-value
    --hip-auto-headers=cuda_open
    -O${optimization_level}
    --cuda-gpu-arch=${mcpu}
    ${CUDA_DEBUG}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src)

#    --hip-auto-headers=cuda

  set(bc1_files)

  foreach(file ${ARGN})
    file(RELATIVE_PATH rfile ${dir} ${file})
    get_filename_component(rdir ${rfile} DIRECTORY)
    get_filename_component(fname ${rfile} NAME_WE)
    get_filename_component(fext ${rfile} EXT)

    set(bc1_filename ${fname}.${mcpu}.bc)

    file(GLOB h_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

    add_custom_command(
      OUTPUT ${bc1_filename}
      COMMAND ${cu_cmd} ${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.cu -o ${bc1_filename}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.cu" ${h_files}
      )

    list(APPEND bc1_files ${bc1_filename})
  endforeach()

  add_custom_command(
    OUTPUT linkout.cuda.${mcpu}.bc
    COMMAND ${AOMP_BINDIR}/llvm-link ${bc1_files} -o linkout.cuda.${mcpu}.bc
    DEPENDS ${bc1_files}
    )

  list(APPEND bc_files linkout.cuda.${mcpu}.bc)
endmacro()

set(libname "omptarget-amdgcn")
collect_sources(${libname} ${CMAKE_CURRENT_SOURCE_DIR} ${sources})

foreach(mcpu ${mcpus})
  set(bc_files)
  add_cuda_bc_library(${libname} ${CMAKE_CURRENT_SOURCE_DIR} ${cuda_sources})

  add_custom_command(
    OUTPUT lib${libname}-${mcpu}.bc
    COMMAND ${AOMP_BINDIR}/llvm-link ${bc_files} -o ${OUTPUTDIR}/lib${libname}-${mcpu}.bc
    DEPENDS ${bc_files}
    )

  add_custom_target(lib${libname}-${mcpu} ALL DEPENDS lib${libname}-${mcpu}.bc)

  install(FILES ${OUTPUTDIR}/lib${libname}-${mcpu}.bc 
     DESTINATION "lib${OPENMP_LIBDIR_SUFFIX}/libdevice"
  )
endforeach()
