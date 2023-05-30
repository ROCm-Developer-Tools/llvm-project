//===-- Passes.td - OpenMP pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TargetOpMapCapturePass : Pass<"targetop-map-capture", "ModuleOp"> {
  let summary = "Populate TargetOp map clause with default captures";
  let description = [{
    Values that are utilised within target operation regions but are 
    defined external to the target region are propogated to map
    clauses so that they can be correctly mapped to devices (or handled
    otherwise correctly). Essentially implicitly capturing these values
    as default map variables. 
  }];
  let constructor = "mlir::omp::createTargetOpMapCapturePass()";
}

#endif // MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES_TD