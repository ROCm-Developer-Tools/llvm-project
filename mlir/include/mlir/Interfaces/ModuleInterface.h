//===-- include/flang/Common/module-wrapper.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_MODULEINTERFACE_H_
#define MLIR_INTERFACES_MODULEINTERFACE_H_

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

// An interface class that contains a single module (which can be more than one
// type) and gives access to a subset of the shared features amongst all
// modules.
class ModuleInterface {
public:
  enum class ModuleType { Builtin, OpenMP, NoModule };

  ModuleInterface() = default;

  ModuleInterface(mlir::omp::ModuleOp mod)
      : module(mod), modType(ModuleType::OpenMP) {}

  ModuleInterface(mlir::omp::ModuleOp *mod)
      : module(*mod), modType(ModuleType::OpenMP) {}

  ModuleInterface(mlir::ModuleOp mod)
      : module(mod), modType(ModuleType::Builtin) {}

  ModuleInterface(mlir::ModuleOp *mod)
      : module(*mod), modType(ModuleType::Builtin) {}

  ModuleInterface(mlir::Operation *mod) : module(mod) {
    if (mlir::isa_and_nonnull<mlir::omp::ModuleOp>(module)) {
      modType = ModuleType::OpenMP;
    } else if (mlir::isa_and_nonnull<mlir::ModuleOp>(module)) {
      modType = ModuleType::Builtin;
    } else {
      modType = ModuleType::NoModule;
    }
  }

  mlir::omp::ModuleOp getAsOMPModule() {
    return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module);
  }

  mlir::ModuleOp getAsBuiltinModule() {
    return mlir::dyn_cast_or_null<mlir::ModuleOp>(module);
  }

  bool isOMPModule() { return (modType == ModuleType::OpenMP); }
  bool isBuiltinModule() { return (modType == ModuleType::Builtin); }

  mlir::Operation *getAsOperation() { return module; }
  ModuleType getModuleType() { return modType; }
  bool isValidModule() {
    if (module &&
        (modType == ModuleType::Builtin || modType == ModuleType::OpenMP)) {
      return true;
    }
    return false;
  }

  // NOTE: This is a bit of an issue as it's used before we know what the module
  // type is that we're handling, however, the module interface is intended to
  // give the base feature set of builtin.module and nothing more, any extra
  // manipulation for specialised types requres appropriate use of getModuleType
  // and getAs*Module
  static ::llvm::StringLiteral getOperationName() { return "builtin.module"; }

  ::mlir::Location getLoc() {
    switch (modType) {
    case ModuleType::Builtin:
      return mlir::dyn_cast_or_null<mlir::ModuleOp>(module).getLoc();
      break;
    case ModuleType::OpenMP:
      return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module).getLoc();
      break;
    default:
      assert(false && "lookupSymbol unsupported for current module type");
      break;
    }
  }

  template <typename FnT>
  auto walk(FnT callback) {
    switch (modType) {
    case ModuleType::Builtin:
      return mlir::dyn_cast_or_null<mlir::ModuleOp>(module)->walk(callback);
      break;
    case ModuleType::OpenMP:
      return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module)->walk(
          callback);
      break;
    default:
      assert(false && "lookupSymbol unsupported for current module type");
      break;
    }
  }

  template <typename T>
  auto getOps() {
    switch (modType) {
    case ModuleType::Builtin:
      return mlir::dyn_cast_or_null<mlir::ModuleOp>(module).getOps<T>();
      break;
    case ModuleType::OpenMP:
      return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module).getOps<T>();
      break;
    default:
      assert(false && "lookupSymbol unsupported for current module type");
      break;
    }
  }

  template <typename T, typename U>
  auto lookupSymbol(U name) {
    switch (modType) {
    case ModuleType::Builtin:
      return mlir::dyn_cast_or_null<mlir::ModuleOp>(module).lookupSymbol<T>(
          name);
      break;
    case ModuleType::OpenMP:
      return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module)
          .lookupSymbol<T>(name);
      break;
    default:
      assert(false && "lookupSymbol unsupported for current module type");
      break;
    }
  }

  std::optional<llvm::StringRef> getName() {
    switch (modType) {
    case ModuleType::Builtin:
      return mlir::dyn_cast_or_null<mlir::ModuleOp>(module).getName();
      break;
    case ModuleType::OpenMP:
      return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module).getName();
      break;
    default:
      assert(false && "getName unsupported for current module type");
      break;
    }
  }

  template <typename PrinterT>
  void print(PrinterT &printer) {
    switch (modType) {
    case ModuleType::Builtin:
      return mlir::dyn_cast_or_null<mlir::ModuleOp>(module).print(printer);
      break;
    case ModuleType::OpenMP:
      return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module).print(printer);
      break;
    default:
      assert(false && "print unsupported for current module type");
      break;
    }
  }

  mlir::Block *getBody() {
    switch (modType) {
    case ModuleType::Builtin:
      return mlir::dyn_cast_or_null<mlir::ModuleOp>(module).getBody();
      break;
    case ModuleType::OpenMP:
      return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module).getBody();
      break;
    default:
      assert(false && "getBody unsupported for current module type");
      break;
    }
  }

  ::mlir::Region &getBodyRegion() {
    switch (modType) {
    case ModuleType::Builtin:
      return mlir::dyn_cast_or_null<mlir::ModuleOp>(module).getBodyRegion();
      break;
    case ModuleType::OpenMP:
      return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module)
          .getBodyRegion();
      break;
    default:
      assert(false && "getBodyRegion unsupported for current module type");
      break;
    }
  }

  mlir::MLIRContext *getContext() {
    switch (modType) {
    case ModuleType::Builtin:
      return mlir::dyn_cast_or_null<mlir::ModuleOp>(module).getContext();
      break;
    case ModuleType::OpenMP:
      return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module).getContext();
      break;
    default:
      assert(false && "getContext unsupported for current module type");
      break;
    }
  }

  template <typename T>
  static bool classof(T op) {
    if constexpr (std::is_same<T, omp::ModuleOp>())
      return true;

    if constexpr (std::is_same<T, ModuleOp>())
      return true;

    if constexpr (std::is_same<T, Operation *>())
      return true;
    return false;
  }

  ::mlir::LogicalResult verifyInvariants() {
    switch (modType) {
    case ModuleType::Builtin:
      return mlir::dyn_cast_or_null<mlir::ModuleOp>(module).verifyInvariants();
      break;
    case ModuleType::OpenMP:
      return mlir::dyn_cast_or_null<mlir::omp::ModuleOp>(module)
          .verifyInvariants();
      break;
    default:
      assert(false && "getContext unsupported for current module type");
      break;
    }
  }

  mlir::Operation *operator->() { return module; }
  operator mlir::Operation *() { return module; }

private:
  mlir::Operation *module = nullptr;
  ModuleType modType = ModuleType::NoModule;
};
} // namespace mlir
#endif // MLIR_INTERFACES_MODULEINTERFACE_H_