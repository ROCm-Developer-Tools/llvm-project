//===-- FIRContext.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Host.h"

void fir::setTargetTriple(std::variant<mlir::ModuleOp, mlir::omp::ModuleOp> mod,
                          llvm::StringRef triple) {
  auto target = fir::determineTargetTriple(triple);
  if (std::holds_alternative<mlir::ModuleOp>(mod)) {
    auto module = std::get<mlir::ModuleOp>(mod);
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    mlir::StringAttr::get(module.getContext(), target));
  } else if (std::holds_alternative<mlir::omp::ModuleOp>(mod)) {
    auto module = std::get<mlir::omp::ModuleOp>(mod);
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    mlir::StringAttr::get(module.getContext(), target));
  }
}

llvm::Triple
fir::getTargetTriple(std::variant<mlir::ModuleOp, mlir::omp::ModuleOp> mod) {
  if (std::holds_alternative<mlir::ModuleOp>(mod)) {
    auto module = std::get<mlir::ModuleOp>(mod);
    if (auto target = module->getAttrOfType<mlir::StringAttr>(
            mlir::LLVM::LLVMDialect::getTargetTripleAttrName()))
      return llvm::Triple(target.getValue());
  }

  if (std::holds_alternative<mlir::omp::ModuleOp>(mod)) {
    auto module = std::get<mlir::omp::ModuleOp>(mod);
    if (auto target = module->getAttrOfType<mlir::StringAttr>(
            mlir::LLVM::LLVMDialect::getTargetTripleAttrName()))
      return llvm::Triple(target.getValue());
  }

  return llvm::Triple(llvm::sys::getDefaultTargetTriple());
}

static constexpr const char *kindMapName = "fir.kindmap";
static constexpr const char *defKindName = "fir.defaultkind";

void fir::setKindMapping(std::variant<mlir::ModuleOp, mlir::omp::ModuleOp> mod,
                         fir::KindMapping &kindMap) {

  if (std::holds_alternative<mlir::ModuleOp>(mod)) {
    auto module = std::get<mlir::ModuleOp>(mod);
    auto *ctx = module.getContext();
    module->setAttr(kindMapName,
                    mlir::StringAttr::get(ctx, kindMap.mapToString()));
    auto defs = kindMap.defaultsToString();
    module->setAttr(defKindName, mlir::StringAttr::get(ctx, defs));

  } else if (std::holds_alternative<mlir::omp::ModuleOp>(mod)) {
    auto module = std::get<mlir::omp::ModuleOp>(mod);
    auto *ctx = module.getContext();
    module->setAttr(kindMapName,
                    mlir::StringAttr::get(ctx, kindMap.mapToString()));
    auto defs = kindMap.defaultsToString();
    module->setAttr(defKindName, mlir::StringAttr::get(ctx, defs));
  }
}

fir::KindMapping
fir::getKindMapping(std::variant<mlir::ModuleOp, mlir::omp::ModuleOp> mod) {
  mlir::MLIRContext *ctx = nullptr;
  if (std::holds_alternative<mlir::ModuleOp>(mod)) {
    auto module = std::get<mlir::ModuleOp>(mod);
    ctx = module.getContext();
    if (auto defs = module->getAttrOfType<mlir::StringAttr>(defKindName)) {
      auto defVals = fir::KindMapping::toDefaultKinds(defs.getValue());
      if (auto maps = module->getAttrOfType<mlir::StringAttr>(kindMapName))
        return fir::KindMapping(ctx, maps.getValue(), defVals);
      return fir::KindMapping(ctx, defVals);
    }
  }

  if (std::holds_alternative<mlir::omp::ModuleOp>(mod)) {
    auto module = std::get<mlir::omp::ModuleOp>(mod);
    ctx = module.getContext();
    if (auto defs = module->getAttrOfType<mlir::StringAttr>(defKindName)) {
      auto defVals = fir::KindMapping::toDefaultKinds(defs.getValue());
      if (auto maps = module->getAttrOfType<mlir::StringAttr>(kindMapName))
        return fir::KindMapping(ctx, maps.getValue(), defVals);
      return fir::KindMapping(ctx, defVals);
    }
  }

  return fir::KindMapping(ctx);
}

std::string fir::determineTargetTriple(llvm::StringRef triple) {
  // Treat "" or "default" as stand-ins for the default machine.
  if (triple.empty() || triple == "default")
    return llvm::sys::getDefaultTargetTriple();
  // Treat "native" as stand-in for the host machine.
  if (triple == "native")
    return llvm::sys::getProcessTriple();
  // TODO: normalize the triple?
  return triple.str();
}
