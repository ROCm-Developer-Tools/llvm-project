
!----------
! RUN lines
!----------
! Embed something that can be easily checked
! RUN: %flang_fc1 -emit-llvm -o - -fembed-offload-object=%S/Inputs/hello.f90 %s 2>&1 | FileCheck %s

! CHECK: llvm.embedded.object
! CHECK: Hello world!

parameter(i=1)
integer :: j
end program
