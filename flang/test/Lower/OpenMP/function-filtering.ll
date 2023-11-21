; ModuleID = 'FIRModule'
source_filename = "FIRModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.__tgt_offload_entry = type { ptr, ptr, i64, i32, i32 }
%struct.__tgt_kernel_arguments = type { i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, [3 x i32], [3 x i32], i32 }

@.__omp_offloading_fd00_47e6f86_target_subr__l44.region_id = weak constant i8 0
@0 = private unnamed_addr constant [106 x i8] c";/work/agozillo/git/trunk17.0/llvm-project/flang/test/Lower/OpenMP/function-filtering.f90;unknown;44;16;;\00", align 1
@.offload_sizes = private unnamed_addr constant [1 x i64] [i64 4]
@.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 34]
@.offload_mapnames = private constant [1 x ptr] [ptr @0]
@1 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @1 }, align 8
@.omp_offloading.entry_name = internal unnamed_addr constant [47 x i8] c"__omp_offloading_fd00_47e6f86_target_subr__l44\00"
@.omp_offloading.entry.__omp_offloading_fd00_47e6f86_target_subr__l44 = weak constant %struct.__tgt_offload_entry { ptr @.__omp_offloading_fd00_47e6f86_target_subr__l44.region_id, ptr @.omp_offloading.entry_name, i64 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1

declare ptr @malloc(i64)

declare void @free(ptr)

define i32 @device_fn_() {
  %1 = alloca i32, i64 1, align 4
  store i32 10, ptr %1, align 4
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}

define i32 @host_fn_() {
  %1 = alloca i32, i64 1, align 4
  store i32 10, ptr %1, align 4
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}

define void @target_subr_(ptr %0) {
  %.offload_baseptrs = alloca [1 x ptr], align 8
  %.offload_ptrs = alloca [1 x ptr], align 8
  %.offload_mappers = alloca [1 x ptr], align 8
  %kernel_args = alloca %struct.__tgt_kernel_arguments, align 8
  br label %entry

entry:                                            ; preds = %1
  %2 = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
  store ptr %0, ptr %2, align 8
  %3 = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
  store ptr %0, ptr %3, align 8
  %4 = getelementptr inbounds [1 x ptr], ptr %.offload_mappers, i64 0, i64 0
  store ptr null, ptr %4, align 8
  %5 = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
  %6 = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
  %7 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 0
  store i32 2, ptr %7, align 4
  %8 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 1
  store i32 1, ptr %8, align 4
  %9 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 2
  store ptr %5, ptr %9, align 8
  %10 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 3
  store ptr %6, ptr %10, align 8
  %11 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 4
  store ptr @.offload_sizes, ptr %11, align 8
  %12 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 5
  store ptr @.offload_maptypes, ptr %12, align 8
  %13 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 6
  store ptr null, ptr %13, align 8
  %14 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 7
  store ptr null, ptr %14, align 8
  %15 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
  store i64 0, ptr %15, align 8
  %16 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 9
  store i64 0, ptr %16, align 8
  %17 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 10
  store [3 x i32] [i32 -1, i32 0, i32 0], ptr %17, align 4
  %18 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 11
  store [3 x i32] zeroinitializer, ptr %18, align 4
  %19 = getelementptr inbounds %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 12
  store i32 0, ptr %19, align 4
  %20 = call i32 @__tgt_target_kernel(ptr @2, i64 -1, i32 -1, i32 0, ptr @.__omp_offloading_fd00_47e6f86_target_subr__l44.region_id, ptr %kernel_args)
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %omp_offload.failed, label %omp_offload.cont

omp_offload.failed:                               ; preds = %entry
  call void @__omp_offloading_fd00_47e6f86_target_subr__l44(ptr %0)
  br label %omp_offload.cont

omp_offload.cont:                                 ; preds = %omp_offload.failed, %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore(ptr) #0

define internal void @__omp_offloading_fd00_47e6f86_target_subr__l44(ptr %0) {
entry:
  br label %omp.target

omp.target:                                       ; preds = %entry
  store i32 10, ptr %0, align 4
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %omp.target
  ret void
}

; Function Attrs: nounwind
declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0, !1}
!omp_offload.info = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 7, !"openmp", i32 11}
!2 = !{i32 0, i32 64768, i32 75394950, !"target_subr_", i32 44, i32 0, i32 0}
