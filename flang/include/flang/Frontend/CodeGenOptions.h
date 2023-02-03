//===--- CodeGenOptions.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CodeGenOptions interface, which holds the
//  configuration for LLVM's middle-end and back-end. It controls LLVM's code
//  generation into assembly or machine code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_CODEGENOPTIONS_H
#define LLVM_CLANG_BASIC_CODEGENOPTIONS_H

#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Regex.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Fortran::frontend {

/// Bitfields of CodeGenOptions, split out from CodeGenOptions to ensure
/// that this large collection of bitfields is a trivial class type.
class CodeGenOptionsBase {

public:
#define CODEGENOPT(Name, Bits, Default) unsigned Name : Bits;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default)
#include "flang/Frontend/CodeGenOptions.def"

protected:
#define CODEGENOPT(Name, Bits, Default)
#define ENUM_CODEGENOPT(Name, Type, Bits, Default) unsigned Name : Bits;
#include "flang/Frontend/CodeGenOptions.def"
};

/// Tracks various options which control how the code is optimized and passed
/// to the LLVM backend.
class CodeGenOptions : public CodeGenOptionsBase {

public:
  struct BitcodeFileToLink {
    /// The filename of the bitcode file to link in.
    std::string Filename;
    /// If true, we set attributes functions in the bitcode library according to
    /// our CodeGenOptions, much as we set attrs on functions that we generate
    /// ourselves.
    bool PropagateAttrs = false;
    /// If true, we use LLVM module internalizer.
    bool Internalize = false;
    /// Bitwise combination of llvm::Linker::Flags, passed to the LLVM linker.
    unsigned LinkFlags = 0;
  };

  /// The files specified here are linked in to the module before optimizations.
  std::vector<BitcodeFileToLink> LinkBitcodeFiles;

  /// The paths to the pass plugins that were registered using -fpass-plugin.
  std::vector<std::string> LLVMPassPlugins;

  /// List of filenames passed in using the -fembed-offload-object option. These
  /// are offloading binaries containing device images and metadata.
  std::vector<std::string> OffloadObjects;

  // Define accessors/mutators for code generation options of enumeration type.
#define CODEGENOPT(Name, Bits, Default)
#define ENUM_CODEGENOPT(Name, Type, Bits, Default)                             \
  Type get##Name() const { return static_cast<Type>(Name); }                   \
  void set##Name(Type Value) { Name = static_cast<unsigned>(Value); }
#include "flang/Frontend/CodeGenOptions.def"

  CodeGenOptions();
};

} // end namespace Fortran::frontend

#endif
