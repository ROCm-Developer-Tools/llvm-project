//===-- lib/Semantics/finalize-omp.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_FINALIZE_OMP_H_
#define FORTRAN_SEMANTICS_FINALIZE_OMP_H_

namespace Fortran::parser {
struct Program;
} // namespace Fortran::parser

namespace Fortran::semantics {
class SemanticsContext;
bool FinalizeOMP(SemanticsContext &context, parser::Program &program);
} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_FINALIZE_OMP_H_
