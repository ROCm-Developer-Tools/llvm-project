!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s 
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-device %s -o - | FileCheck %s

module test_0
    implicit none

!CHECK-DAG: fir.global @_QMtest_0Edata_int {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32
INTEGER :: data_int = 10
!$omp declare target link(data_int)

!CHECK-DAG: fir.global @_QMtest_0Earray_1d(dense<[1, 2, 3]> : tensor<3xi32>) {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !fir.array<3xi32>
INTEGER :: array_1d(3) = (/1,2,3/)
!$omp declare target link(array_1d)

!CHECK-DAG: fir.global @_QMtest_0Earray_2d {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !fir.array<2x2xi32> 
INTEGER :: array_2d(2,2) = reshape((/1,2,3,4/), (/2,2/))
!$omp declare target link(array_2d)

!CHECK-DAG: fir.global @_QMtest_0Ept1 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !fir.box<!fir.ptr<i32>>
INTEGER, POINTER :: pt1
!$omp declare target link(pt1)

!CHECK-DAG: fir.global @_QMtest_0Ept2_tar {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} target : i32
INTEGER, TARGET :: pt2_tar = 5 
!$omp declare target link(pt2_tar)

!CHECK-DAG: fir.global @_QMtest_0Ept2 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !fir.box<!fir.ptr<i32>>
INTEGER, POINTER :: pt2 => pt2_tar
!$omp declare target link(pt2)

!CHECK-DAG: fir.global @_QMtest_0Edata_int_to {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32
INTEGER :: data_int_to = 5
!$omp declare target to(data_int_to)

!CHECK-DAG: fir.global @_QMtest_0Edata_int_clauseless {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32
INTEGER :: data_int_clauseless = 1
!$omp declare target(data_int_clauseless)

!CHECK-DAG: fir.global @_QMtest_0Edata_extended_to_1 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : f32
!CHECK-DAG: fir.global @_QMtest_0Edata_extended_to_2 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : f32
REAL :: data_extended_to_1 = 2
REAL :: data_extended_to_2 = 3
!$omp declare target to(data_extended_to_1, data_extended_to_2)

!CHECK-DAG: fir.global @_QMtest_0Edata_extended_link_1 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : f32
!CHECK-DAG: fir.global @_QMtest_0Edata_extended_link_2 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : f32
REAL :: data_extended_link_1 = 2
REAL :: data_extended_link_2 = 3
!$omp declare target link(data_extended_link_1, data_extended_link_2)

contains
end module test_0
