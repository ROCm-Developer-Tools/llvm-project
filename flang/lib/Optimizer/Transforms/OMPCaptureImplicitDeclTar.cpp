#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>

namespace fir {
#define GEN_PASS_DEF_CAPTUREIMPLICITLYDECLARETARGETPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class CaptureImplicitlyDeclareTargetPass
    : public fir::impl::CaptureImplicitlyDeclareTargetPassBase<
          CaptureImplicitlyDeclareTargetPass> {

  void markNestedFuncs(mlir::func::FuncOp functionOp, mlir::ModuleOp moduleOp) {
    for (auto callOp : functionOp.getOps<fir::CallOp>()) {
      if (auto currFOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(
              callOp.getCallee().value())) {
        auto parentDt =
            mlir::dyn_cast<mlir::omp::DeclareTargetInterface>(*functionOp)
                .getDeclareTargetDeviceType();
        if (mlir::dyn_cast<mlir::omp::DeclareTargetInterface>(*currFOp)
                .isDeclareTarget()) {
          auto currentDt =
              mlir::dyn_cast<mlir::omp::DeclareTargetInterface>(*currFOp)
                  .getDeclareTargetDeviceType();
          // Found the same function twice, with different device_types, mark as
          // Any as it belongs to both
          if (currentDt != parentDt &&
              currentDt != mlir::omp::DeclareTargetDeviceType::any) {
            mlir::dyn_cast<mlir::omp::DeclareTargetInterface>(*currFOp)
                .setDeclareTarget(
                    mlir::omp::DeclareTargetDeviceType::any,
                    mlir::dyn_cast<mlir::omp::DeclareTargetInterface>(
                        *functionOp)
                        .getDeclareTargetCaptureClause());
          }
        } else {
          mlir::dyn_cast<mlir::omp::DeclareTargetInterface>(*currFOp)
              .setDeclareTarget(
                  parentDt,
                  mlir::dyn_cast<mlir::omp::DeclareTargetInterface>(*functionOp)
                      .getDeclareTargetCaptureClause());
        }

        markNestedFuncs(currFOp, moduleOp);
      }
    }
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    for (auto functionOp : moduleOp.getOps<mlir::func::FuncOp>()) {
      auto declareTargetOp =
          llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(*functionOp);
      if (declareTargetOp.isDeclareTarget()) {
        markNestedFuncs(functionOp, moduleOp);
      }
    }
  }
};

} // namespace

namespace fir {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCaptureImplicitlyDeclareTargetPass() {
  return std::make_unique<CaptureImplicitlyDeclareTargetPass>();
}
} // namespace fir
