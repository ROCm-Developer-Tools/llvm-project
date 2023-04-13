!RUN: %flang_fc1 -emit-llvm-bc -fopenmp -o %t.bc %s | llvm-dis %t.bc -o - | FileCheck %s --check-prefix=CHECK-HOST
!RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-is-device -fopenmp-host-ir-file-path %t.bc -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE

! Temporary test for automating checking the lowering and 
! showing what's available at the moment. It needs to be 
! broken into multiple as Flang-new has no location to 
! test from end to end that I can find e.g. fortran all the 
! way down to LLVM-IR

! NOTE: When Clang is used with just -fopenmp-is-device, the metadata 
! for the device is not generated, but with --offload-device-only, it
! goes through extra steps to generate it.

! NOTE: * Missing attributes on globals such as align and dso_local, and protected and 
!         address space (1) on device when using the "To" clause
!       * Pointers look very different in the IR, unsure if this is just a fortran thing
!       * There is some embedded object metadata injected depending on the flags used, but
!         this seems like another unrelated component/task

! NOTE: Anything in triple exclamation marks doesn't exist at the moment and
! needs to be worked on

!! LLVM generated once for the module at entry, dependent on host or device
!CHECK-HOST-DAG: %struct.__tgt_offload_entry = type { ptr, ptr, i64, i32, i32 }
!CHECK-HOST-DAG: !omp_offload.info = !{!{{.*}}}

!!!CHECK-HOST-DAG @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @.omp_offloading.requires_reg, ptr null }]
!!!CHECK-HOST-LABEL define internal void @.omp_offloading.requires_reg()
!!!CHECK-HOST-LABEL declare void @__tgt_register_requires(i64)

!CHECK-DEVICE-DAG: !omp_offload.info = !{!{{.*}}}

module test_0
    implicit none

!CHECK-DEVICE-DAG: @_QMtest_0Edata_int_decl_tgt_ref_ptr = weak global ptr null
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}

!CHECK-HOST-DAG: @_QMtest_0Edata_int = global i32 1
!CHECK-HOST-DAG: @_QMtest_0Edata_int_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Edata_int 
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [36 x i8] c"_QMtest_0Edata_int_decl_tgt_ref_ptr\00"
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Edata_int_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_int_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
INTEGER :: data_int = 10
!$omp declare target link(data_int)

!CHECK-DEVICE-DAG: @_QMtest_0Earray_1d_decl_tgt_ref_ptr = weak global ptr null
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Earray_1d_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}

!CHECK-HOST-DAG: @_QMtest_0Earray_1d = global [3 x i32] [i32 1, i32 2, i32 3]
!CHECK-HOST-DAG: @_QMtest_0Earray_1d_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Earray_1d 
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [36 x i8] c"_QMtest_0Earray_1d_decl_tgt_ref_ptr\00"
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Earray_1d_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Earray_1d_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Earray_1d_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
INTEGER :: array_1d(3) = (/1,2,3/)
!$omp declare target link(array_1d)

!CHECK-DEVICE-DAG: @_QMtest_0Earray_2d_decl_tgt_ref_ptr = weak global ptr null
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Earray_2d_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}

!CHECK-HOST-DAG: @_QMtest_0Earray_2d = global [2 x [2 x i32]] {{.*}}
!CHECK-HOST-DAG: @_QMtest_0Earray_2d_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Earray_2d 
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [36 x i8] c"_QMtest_0Earray_2d_decl_tgt_ref_ptr\00"
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Earray_2d_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Earray_2d_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Earray_2d_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
INTEGER :: array_2d(2,2) = reshape((/1,2,3,4/), (/2,2/))
!$omp declare target link(array_2d)

!CHECK-DEVICE-DAG: @_QMtest_0Ept1_decl_tgt_ref_ptr = weak global ptr null
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Ept1_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}

!CHECK-HOST-DAG: @_QMtest_0Ept1 = global { ptr, i64, i32, i8, i8, i8, i8 } { ptr null, i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 9, i8 1, i8 0 }
!CHECK-HOST-DAG: @_QMtest_0Ept1_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Ept1
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [31 x i8] c"_QMtest_0Ept1_decl_tgt_ref_ptr\00"
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Ept1_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Ept1_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Ept1_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
INTEGER, POINTER :: pt1
!$omp declare target link(pt1)

!CHECK-DEVICE-DAG: @_QMtest_0Ept2_tar_decl_tgt_ref_ptr = weak global ptr null
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Ept2_tar_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}

!CHECK-HOST-DAG: @_QMtest_0Ept2_tar = global i32 5
!CHECK-HOST-DAG: @_QMtest_0Ept2_tar_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Ept2_tar
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [35 x i8] c"_QMtest_0Ept2_tar_decl_tgt_ref_ptr\00"
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Ept2_tar_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Ept2_tar_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Ept2_tar_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
INTEGER, TARGET :: pt2_tar = 5 
!$omp declare target link(pt2_tar)

!CHECK-DEVICE-DAG: @_QMtest_0Ept2_decl_tgt_ref_ptr = weak global ptr null
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Ept2_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}

!CHECK-HOST-DAG: @_QMtest_0Ept2 = global { ptr, i64, i32, i8, i8, i8, i8 } { ptr @_QMtest_0Ept2_tar, i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 9, i8 1, i8 0 }
!CHECK-HOST-DAG: @_QMtest_0Ept2_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Ept2
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [31 x i8] c"_QMtest_0Ept2_decl_tgt_ref_ptr\00"
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Ept2_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Ept2_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Ept2_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
INTEGER, POINTER :: pt2 => pt2_tar
!$omp declare target link(pt2)

!CHECK-DEVICE-DAG: @_QMtest_0Edata_int_to = global i32 5
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_to", i32 {{.*}}, i32 {{.*}}}

!CHECK-HOST-DAG: @_QMtest_0Edata_int_to = global i32 5
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [22 x i8] c"_QMtest_0Edata_int_to\00"
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Edata_int_to = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_int_to, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_to", i32 {{.*}}, i32 {{.*}}}
INTEGER :: data_int_to = 5
!$omp declare target to(data_int_to)

!CHECK-DEVICE-DAG: @_QMtest_0Edata_int_clauseless = global i32 1
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_clauseless", i32 {{.*}}, i32 {{.*}}}

!CHECK-HOST-DAG: @_QMtest_0Edata_int_clauseless = global i32 1
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [30 x i8] c"_QMtest_0Edata_int_clauseless\00"
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Edata_int_clauseless = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_int_clauseless, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_clauseless", i32 {{.*}}, i32 {{.*}}}
INTEGER :: data_int_clauseless = 1
!$omp declare target(data_int_clauseless)

!CHECK-DEVICE-DAG: @_QMtest_0Edata_extended_to_1 = global float 2.000000e+00
!CHECK-DEVICE-DAG: @_QMtest_0Edata_extended_to_2 = global float 3.000000e+00
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_to_1", i32 {{.*}}, i32 {{.*}}}
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_to_2", i32 {{.*}}, i32 {{.*}}}

!CHECK-HOST-DAG: @_QMtest_0Edata_extended_to_1 = global float 2.000000e+00
!CHECK-HOST-DAG: @_QMtest_0Edata_extended_to_2 = global float 3.000000e+00
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [29 x i8] c"_QMtest_0Edata_extended_to_1\00"
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [29 x i8] c"_QMtest_0Edata_extended_to_2\00"
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Edata_extended_to_1 = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_extended_to_1, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Edata_extended_to_2 = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_extended_to_2, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_to_1", i32 {{.*}}, i32 {{.*}}}
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_to_2", i32 {{.*}}, i32 {{.*}}}
REAL :: data_extended_to_1 = 2
REAL :: data_extended_to_2 = 3
!$omp declare target to(data_extended_to_1, data_extended_to_2)


!CHECK-DEVICE-DAG: @_QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr = weak global ptr null
!CHECK-DEVICE-DAG: @_QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr = weak global ptr null
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
!CHECK-DEVICE-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}


!CHECK-HOST-DAG: @_QMtest_0Edata_extended_link_1 = global float 2.000000e+00
!CHECK-HOST-DAG: @_QMtest_0Edata_extended_link_2 = global float 3.000000e+00
!CHECK-HOST-DAG: @_QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Edata_extended_link_1
!CHECK-HOST-DAG: @_QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Edata_extended_link_2
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [48 x i8] c"_QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr\00"
!CHECK-HOST-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [48 x i8] c"_QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr\00"
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: @.omp_offloading.entry._QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
!CHECK-HOST-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
REAL :: data_extended_link_1 = 2
REAL :: data_extended_link_2 = 3
!$omp declare target link(data_extended_link_1, data_extended_link_2)

contains
end module test_0