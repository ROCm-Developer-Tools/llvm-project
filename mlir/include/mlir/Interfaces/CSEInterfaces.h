//===- CSEInterfaces.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_INTERFACES_CSEINTERFACES_H_
#define MLIR_INTERFACES_CSEINTERFACES_H_

#include "mlir/IR/DialectInterface.h"

namespace mlir {
class Operation;

/// Define an interface to allow for dialects to control specific aspects of
/// common subexpression elimination behavior for operations they define.
class DialectCSEInterface : public DialectInterface::Base<DialectCSEInterface> {
public:
  DialectCSEInterface(Dialect *dialect) : Base(dialect) {}

  /// Registered hook to check if an operation that is *not* isolated from
  /// above, should allow common subexpressions to be extracted out of its
  /// regions.
  virtual bool subexpressionExtractionAllowed(Operation *op) const {
    return true;
  }
};

} // namespace mlir

#endif // MLIR_INTERFACES_CSEINTERFACES_H_
