! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck --check-prefix=MLIR %s
! RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck --check-prefix=LLVM %s 

! This test checks the lowering of requires into MLIR and LLVM IR

! MLIR attributes
!MLIR:      module attributes {
!MLIR-SAME: omp.atomic_default_mem_order = #omp<memoryorderkind seq_cst>
!MLIR-SAME: omp.requires = #omp<clause_requires reverse_offload|unified_shared_memory>

! Global constructors
!LLVM:      @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
!LLVM-SAME: [{ i32, ptr, ptr } { i32 0, ptr @.omp_offloading.requires_reg, ptr null }]

! Atomic memory order matches default set by requires directive
!LLVM-LABEL: define void @_QQmain() {
!LLVM:         load atomic
!LLVM-SAME:    seq_cst

! Definition of global constructor, sending the proper flags to the runtime
!LLVM-LABEL: define internal void @.omp_offloading.requires_reg()
!LLVM-SAME:  #[[OFFLOAD_ATTR:[0-9]+]] section ".text.startup" {
!LLVM-NEXT:  entry:
!LLVM-NEXT:    call void @__tgt_register_requires(i64 10)
!LLVM-NEXT:    ret void
!LLVM-NEXT:  }

!LLVM: attributes #[[OFFLOAD_ATTR]] = { noinline nounwind }

program requires
  !$omp requires unified_shared_memory reverse_offload atomic_default_mem_order(seq_cst)
  integer :: x, y

  !$omp atomic read
  x = y
end program requires
