//===- TargetOpMapCapture.cpp - Capture Implicits in TargetOp Map Clause --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/IR/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/Frontend/OpenMP/OMPConstants.h"

namespace mlir {
namespace {

#define GEN_PASS_DEF_TARGETOPMAPCAPTUREPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"

struct TargetOpMapCapturePass
    : public impl::TargetOpMapCapturePassBase<
          TargetOpMapCapturePass> {

  using TargetOpMapCapturePassBase<
      TargetOpMapCapturePass>::TargetOpMapCapturePassBase;

  void runOnOperation() override {
    auto module = getOperation();

    module.walk([&](mlir::omp::TargetOp tarOp) {
        llvm::SetVector<Value> operandSet;
        getUsedValuesDefinedAbove(tarOp.getRegion(), operandSet);
        
        llvm::SmallVector<mlir::Value> usedButNotCaptured;
        for (auto v : operandSet) {
            bool insertable = true;
            for (auto mapOp : tarOp.getMapOperands())
              if (v == mapOp)
               insertable = false;
            
            if (insertable)
              usedButNotCaptured.push_back(v);
        }

        llvm::SmallVector<mlir::Attribute> newMapTypesAttr(tarOp.getMapTypesAttr().begin(),
                                                        tarOp.getMapTypesAttr().end());

        // NOTE: Ponter-case, unused currently as it is a WIP.
        llvm::omp::OpenMPOffloadMappingFlags captureByThis =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;

        llvm::omp::OpenMPOffloadMappingFlags literalCapture =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_LITERAL;

        llvm::omp::OpenMPOffloadMappingFlags mapTypeBits;
        // Mimicing Map Type Generation code from CGOpenMPRuntime.cpp in Clang's
        // generateDefaultMapInfo, this is an initial 
        for (unsigned i = 0; i < usedButNotCaptured.size(); ++i) {
            mapTypeBits = llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE;

            // TODO: Case for pointers/non-literals
            mapTypeBits = literalCapture;
            
            // All captures are target_param
            mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::
                        OMP_MAP_TARGET_PARAM;

            // TODO: not all captures are implicit, but it is the default case
            // handling this needs to be extended to handle the non-default 
            mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;

            newMapTypesAttr.push_back(IntegerAttr::get(
                IntegerType::get(module.getContext(), 64),
                static_cast<std::underlying_type_t<
                    llvm::omp::OpenMPOffloadMappingFlags>>(
                    mapTypeBits)));
        }

        tarOp.getMapOperandsMutable().append(usedButNotCaptured);
        tarOp.setMapTypesAttr(ArrayAttr::get(module.getContext(), newMapTypesAttr));
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::omp::createTargetOpMapCapturePass() {
  return std::make_unique<TargetOpMapCapturePass>();
}

} // namespace mlir

