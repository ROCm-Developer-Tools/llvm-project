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
// TODO: Handle case where a function is marked twice by declare target and it's
// two different target types
class CaptureImplicitlyDeclareTargetPass
    : public fir::impl::CaptureImplicitlyDeclareTargetPassBase<
          CaptureImplicitlyDeclareTargetPass> {

  // TODO: deal with finding the same function twice, with different device_type
  // should be combined into Any, or Any should supersede what was before
  void markNestedFuncs(mlir::func::FuncOp functionOp,
                       mlir::Operation *moduleOp) {
    llvm::errs() << "CurrentFuncName: " << functionOp.getName() << "\n";
    for (auto callOp : functionOp.getOps<fir::CallOp>()) {
      auto currFOp = llvm::dyn_cast_or_null<mlir::func::FuncOp>(
          mlir::SymbolTable::lookupSymbolIn(moduleOp,
                                            callOp.getCallee().value()));
      if (currFOp) {
        mlir::omp::OpenMPDialect::setDeclareTarget(
            currFOp,
            mlir::omp::OpenMPDialect::getDeclareTargetDeviceType(functionOp));
        markNestedFuncs(currFOp, moduleOp);
      }
    }
  }

  void runOnOperation() override {
    mlir::func::FuncOp functionOp = getOperation();
    if (mlir::omp::OpenMPDialect::isDeclareTarget(functionOp)) {
      auto *moduleOp =
          functionOp->getParentWithTrait<mlir::OpTrait::SymbolTable>();
      assert(moduleOp && "Expected function to be inside a module");
      markNestedFuncs(functionOp, moduleOp);
    }
  }
};

} // namespace

namespace fir {
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createCaptureImplicitlyDeclareTargetPass() {
  return std::make_unique<CaptureImplicitlyDeclareTargetPass>();
}
} // namespace fir
