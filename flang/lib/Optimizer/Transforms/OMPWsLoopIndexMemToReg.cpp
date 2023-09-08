//===- OMPWsLoopIndexMem2Reg.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to remove `fir.alloca`s, together with their
// associated `fir.load` and `fir.store` operations, when they are only used to
// store the index variable of an `omp.wsloop` and never passed by reference.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/Passes.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

namespace fir {
#define GEN_PASS_DEF_OMPWSLOOPINDEXMEMTOREG
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {
class OMPWsLoopIndexMemToRegPass
    : public fir::impl::OMPWsLoopIndexMemToRegBase<OMPWsLoopIndexMemToRegPass> {
public:
  void runOnOperation() override {
    llvm::SmallPtrSet<Operation *, 4> opsToDelete;
    func::FuncOp func = getOperation();

    func.walk([&](omp::WsLoopOp loop) {
      llvm::SmallVector<llvm::SmallVector<Value>> storeAddresses;
      llvm::ArrayRef<BlockArgument> loopArgs = loop.getRegion().getArguments();

      // Collect indices of the omp.wsloop operation.
      for (BlockArgument arg : loopArgs) {
        // Find fir.store uses of these indices and gather all addresses where
        // they are stored.
        llvm::SmallVector<Value> &argStores = storeAddresses.emplace_back();
        for (OpOperand &argUse : arg.getUses())
          if (auto storeOp = dyn_cast<fir::StoreOp>(argUse.getOwner()))
            argStores.push_back(storeOp.getMemref());
      }

      auto processArg = [&](BlockArgument arg,
                            llvm::ArrayRef<Value> argStores) {
        llvm::SmallPtrSet<Operation *, 4> toDelete;
        for (Value store : argStores) {
          // Skip argument if storage not defined by a fir.alloca.
          if (!isa_and_nonnull<fir::AllocaOp>(store.getDefiningOp()))
            return;

          // Check that uses of the pointer are all fir.load and fir.store
          // inside of the omp.wsloop currently being visited.
          bool patternApplicable = true;
          for (OpOperand &use : store.getUses()) {
            Operation *owner = use.getOwner();
            if (owner->getParentOfType<omp::WsLoopOp>() !=
                    loop.getOperation() ||
                (!isa<fir::LoadOp>(owner) && !isa<fir::StoreOp>(owner))) {
              patternApplicable = false;
              break;
            }
          }

          // Do not make any modifications if some uses of the pointer are
          // outside of the omp.wsloop.
          if (!patternApplicable)
            return;

          // Remove fir.store operations for that address and replace all
          // fir.load operations with the index as returned by the omp.wsloop
          // operation.
          for (OpOperand &use : store.getUses()) {
            Operation *owner = use.getOwner();
            if (isa<fir::StoreOp>(owner))
              toDelete.insert(owner);
            else if (isa<fir::LoadOp>(owner))
              toDelete.insert(owner);
          }

          // Delete now-unused fir.alloca.
          toDelete.insert(store.getDefiningOp());
          store.dropAllUses();
        }

        // Only consider marked operations if all fir.{load,store,alloca}
        // operations associated with the given loop index can be removed.
        opsToDelete.insert(toDelete.begin(), toDelete.end());

        for (Operation *op : toDelete)
          if (isa<fir::LoadOp>(op))
            op->replaceAllUsesWith(ValueRange(arg));
      };

      for (auto it : llvm::zip(loopArgs, storeAddresses))
        processArg(std::get<0>(it), std::get<1>(it));
    });

    // Delete marked operations.
    for (Operation *op : opsToDelete)
      op->erase();
  }
};
} // namespace

std::unique_ptr<Pass> fir::createOMPWsLoopIndexMemToRegPass() {
  return std::make_unique<OMPWsLoopIndexMemToRegPass>();
}
