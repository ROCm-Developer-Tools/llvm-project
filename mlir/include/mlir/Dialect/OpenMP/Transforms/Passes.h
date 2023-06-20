//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the Func
// transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::omp {

#define GEN_PASS_DECL
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"

/// Create default map clause entries for target operations with implicit captures
std::unique_ptr<Pass> createTargetOpMapCapturePass();

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"

}

#endif // MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES_H