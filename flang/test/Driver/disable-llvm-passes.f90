! Ensure argument -disable-llvm-passes works as expected.

! RUN: %flang_fc1 -O2 -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=PASSES
! RUN: %flang_fc1 -O2 -disable-llvm-passes -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=NOPASSES

! PASSES: define void @f
! PASSES-NOT: store float 1.0

! NOPASSES: define void @f
! NOPASSES: store float 1.0

subroutine f()
  ! x is not used, it can be optimized out
  real :: x
  x = 1.0
end subroutine
