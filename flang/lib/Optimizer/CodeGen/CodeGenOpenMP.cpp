#include "flang/Optimizer/CodeGen/CodeGen.h"

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_OPENMPFIRCONVERSIONSTOLLVM
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"
} // namespace fir

using namespace fir;

#define DEBUG_TYPE "flang-codegen-openmp"

// fir::LLVMTypeConverter for converting to LLVM IR dialect types.
#include "flang/Optimizer/CodeGen/TypeConverter.h"

namespace {
class OpenMPFIRConversionsToLLVM
    : public fir::impl::OpenMPFIRConversionsToLLVMBase<
          OpenMPFIRConversionsToLLVM> {
public:
  OpenMPFIRConversionsToLLVM() {}

  inline mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    fir::LLVMTypeConverter typeConverter{getModule(), /*applyTBAA*/ false,
                                         /*forceUnifiedTBAATree*/ false};
    mlir::IRRewriter rewriter(getModule()->getContext());
    getModule().walk([&](mlir::Operation *op) {
      // FIR Op specific conversion for MapInfoOp's containing BoxTypes that are
      // descriptors this allows FIR specific lowering of types, required for
      // descriptors of allocatables currently.
      if (auto mapInfoOp = mlir::dyn_cast<mlir::omp::MapInfoOp>(op)) {
        if (mapInfoOp.getIsFortranAllocatable() &&
            mapInfoOp.getVarType().has_value()) {
          mapInfoOp.setVarType(typeConverter.convertBoxTypeAsStruct(
              mlir::cast<fir::BaseBoxType>(mapInfoOp.getVarType().value())));
        }
      }
    });
  };
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createOpenMPFIRConversionsToLLVMPass() {
  return std::make_unique<OpenMPFIRConversionsToLLVM>();
}
