// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_split.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-LOOP %S/base_split.h
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-6

#define SCHEDULE runtime
#include "base_split.h"
