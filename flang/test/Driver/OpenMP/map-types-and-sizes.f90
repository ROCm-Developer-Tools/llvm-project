!RUN: %flang_fc1 -emit-llvm -fopenmp -flang-deprecated-no-hlfir %s -o - | FileCheck %s

!===============================================================================
! Check MapTypes for target implicit captures
!===============================================================================

!CHECK: @.offload_sizes = private unnamed_addr constant [1 x i64] [i64 4]
!CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 800]
subroutine mapType_scalar
  integer :: a
  !$omp target
     a = 10
  !$omp end target
end subroutine mapType_scalar

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 4096]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 547]
subroutine mapType_array
  integer :: a(1024)
  !$omp target
     a(10) = 20
  !$omp end target
end subroutine mapType_array

!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 547]
subroutine mapType_ptr
  integer, pointer :: a
  !$omp target
     a = 10
  !$omp end target
end subroutine mapType_ptr

!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 547]
subroutine mapType_allocatable
  integer, allocatable :: a
  allocate(a)
  !$omp target
     a = 10
  !$omp end target
  deallocate(a)
end subroutine mapType_allocatable

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 8, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 544, i64 800]
subroutine mapType_c_ptr
  use iso_c_binding, only : c_ptr, c_loc
  type(c_ptr) :: a
  integer, target :: b
  !$omp target
     a = c_loc(b)
  !$omp end target
end subroutine mapType_c_ptr

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 1]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 800]
subroutine mapType_char
  character :: a
  !$omp target
     a = 'b'
  !$omp end target
end subroutine mapType_char

!CHECK-LABEL: define void @maptype_ptr_() {
!CHECK: %[[DESC_ELE_SIZE:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %{{.*}}, i64 0, i32 1
!CHECK: %[[DESC_ELE_SIZE_LOAD:.*]] = load i64, ptr %[[DESC_ELE_SIZE]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [1 x i64], ptr %.offload_sizes, i32 0, i32 0  
!CHECK: store i64 %[[DESC_ELE_SIZE_LOAD]], ptr %[[OFFLOAD_SIZE_ARR]], align 8

!CHECK-LABEL: define void @maptype_allocatable_() {
!CHECK: %[[DESC_ELE_SIZE:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %{{.*}}, i64 0, i32 1
!CHECK: %[[DESC_ELE_SIZE_LOAD:.*]] = load i64, ptr %[[DESC_ELE_SIZE]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR_1:.*]] = getelementptr inbounds [1 x i64], ptr %.offload_sizes, i32 0, i32 0  
!CHECK: store i64 %[[DESC_ELE_SIZE_LOAD]], ptr %[[OFFLOAD_SIZE_ARR_1]], align 8