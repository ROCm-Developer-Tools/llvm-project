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
#include <cstdint>

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

    llvm::errs() << "enter \n";
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

        llvm::errs() << "Upper Before \n";
        tarOp.getMapUpperBoundSegmentAttr().dump();
        for (unsigned int i = 0;
             i < tarOp.getMapRangeUpperBoundMutable().size(); ++i) {
            for (unsigned int j = 0;
                 j < tarOp.getMapRangeUpperBoundMutable()[i].size(); ++j) {
              tarOp.getMapRangeUpperBoundMutable()[i][j].dump();
            }
        }

        llvm::errs() << "Lower Before \n";
        tarOp.getMapLowerBoundSegmentAttr().dump();
        for (unsigned int i = 0;
             i < tarOp.getMapRangeLowerBoundMutable().size(); ++i) {
            for (unsigned int j = 0;
                 j < tarOp.getMapRangeLowerBoundMutable()[i].size(); ++j) {
              tarOp.getMapRangeLowerBoundMutable()[i][j].dump();
            }
        }

        auto lb = tarOp.getMapRangeLowerBoundMutable().getBase().first;
        mlir::MutableOperandRange lowerOpRange(std::move(lb));
        std::vector<int32_t> lowerSegment;
        for (unsigned int i = 0;
             i < tarOp.getMapRangeLowerBoundMutable().size(); ++i)
            lowerSegment.push_back(
                tarOp.getMapRangeLowerBoundMutable()[i].size());

        auto ub = tarOp.getMapRangeUpperBoundMutable().getBase().first;
        mlir::MutableOperandRange upperOpRange(std::move(ub));
        std::vector<int32_t> upperSegment;
        for (unsigned int i = 0;
             i < tarOp.getMapRangeUpperBoundMutable().size(); ++i)
            upperSegment.push_back(
                tarOp.getMapRangeUpperBoundMutable()[i].size());

        llvm::SmallVector<mlir::Attribute> newMapTypesAttr(tarOp.getMapTypesAttr().begin(),
                                                        tarOp.getMapTypesAttr().end());

        llvm::SmallVector<mlir::Attribute> newMapCapturesAttr(
            tarOp.getMapCaptureTypesAttr().begin(),
            tarOp.getMapCaptureTypesAttr().end());

        // NOTE: Ponter-case, unused currently as it is a WIP.
        // llvm::omp::OpenMPOffloadMappingFlags captureByThis =
        //     llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
        //     llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;

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

            // TODO: placeholder, this implicit capture likely has to be moved
            // further up into the PFT -> MLIR lowering to get access to more
            // information to decide what is ByThis/ByVal etc.
            newMapCapturesAttr.push_back(
                mlir::omp::VariableCaptureKindAttr::get(
                    module.getContext(),
                    mlir::omp::VariableCaptureKind::ByCopy));

            // implicit map has no range, but we fill in empty data for
            // consistency in the later lowering step.
            upperSegment.push_back(0);
            lowerSegment.push_back(0);
            upperOpRange.append(mlir::ValueRange{});
            lowerOpRange.append(mlir::ValueRange{});
        }

        tarOp.getMapOperandsMutable().append(usedButNotCaptured);
        tarOp.setMapTypesAttr(
            ArrayAttr::get(module.getContext(), newMapTypesAttr));
        tarOp.setMapCaptureTypesAttr(ArrayAttr::get(module.getContext(), newMapCapturesAttr));

        tarOp.setMapLowerBoundSegmentAttr(mlir::DenseI32ArrayAttr::get(
            module->getContext(), llvm::ArrayRef<int32_t>{lowerSegment}));
        tarOp.getMapRangeLowerBoundMutable() = mlir::MutableOperandRangeRange(
            lowerOpRange,
            mlir::NamedAttribute(tarOp.getMapLowerBoundSegmentAttrName(),
                                 tarOp.getMapLowerBoundSegmentAttr()));

        tarOp.setMapUpperBoundSegmentAttr(mlir::DenseI32ArrayAttr::get(
            module->getContext(), llvm::ArrayRef<int32_t>{upperSegment}));
        tarOp.getMapRangeUpperBoundMutable() = mlir::MutableOperandRangeRange(
            upperOpRange,
            mlir::NamedAttribute(tarOp.getMapUpperBoundSegmentAttrName(),
                                 tarOp.getMapUpperBoundSegmentAttr()));

        llvm::errs() << "Upper After \n";
        tarOp.getMapUpperBoundSegmentAttr().dump();
        for (unsigned int i = 0;
             i < tarOp.getMapRangeUpperBoundMutable().size(); ++i) {
            for (unsigned int j = 0;
                 j < tarOp.getMapRangeUpperBoundMutable()[i].size(); ++j) {
              tarOp.getMapRangeUpperBoundMutable()[i][j].dump();
            }
        }

        llvm::errs() << "Lower After \n";
        tarOp.getMapLowerBoundSegmentAttr().dump();
        for (unsigned int i = 0;
             i < tarOp.getMapRangeLowerBoundMutable().size(); ++i) {
            for (unsigned int j = 0;
                 j < tarOp.getMapRangeLowerBoundMutable()[i].size(); ++j) {
              tarOp.getMapRangeLowerBoundMutable()[i][j].dump();
            }
        }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::omp::createTargetOpMapCapturePass() {
  return std::make_unique<TargetOpMapCapturePass>();
}

} // namespace mlir

