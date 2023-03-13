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
            mlir::omp::OpenMPDialect::getDeclareTargetDeviceType(functionOp);
        if (mlir::omp::OpenMPDialect::isDeclareTarget(currFOp)) {
          auto currentDt =
              mlir::omp::OpenMPDialect::getDeclareTargetDeviceType(currFOp);

          // Found the same function twice, with different device_types, mark as
          // Any as it belongs to both
          if (currentDt != parentDt &&
              currentDt != mlir::omp::DeclareTargetDeviceType::any) {
            mlir::omp::OpenMPDialect::setDeclareTarget(
                currFOp, mlir::omp::DeclareTargetDeviceType::any);
          }
        } else {
          mlir::omp::OpenMPDialect::setDeclareTarget(currFOp, parentDt);
        }

        markNestedFuncs(currFOp, moduleOp);
      }
    }
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    for (auto functionOp : moduleOp.getOps<mlir::func::FuncOp>()) {
      if (mlir::omp::OpenMPDialect::isDeclareTarget(functionOp)) {
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
