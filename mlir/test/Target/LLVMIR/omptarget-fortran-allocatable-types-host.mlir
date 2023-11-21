// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks the offload sizes, map types and base pointers and pointers
// provided to the OpenMP kernel argument structure are correct when lowering 
// to LLVM-IR from MLIR when the fortran allocatables flag is switched on and 
// a fortran allocatable descriptor type is provided alongside the omp.map_info,
// the test utilises mapping of array sections, full arrays and individual 
// allocated scalars.

module attributes {omp.is_target_device = false} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @_QQmain() attributes {bindc_name = "main"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.mlir.constant(10 : index) : i64
    %8 = llvm.mlir.constant(5 : index) : i64
    %9 = llvm.mlir.constant(2 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.mlir.addressof @_QFEfull_arr : !llvm.ptr
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.alloca %13 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {bindc_name = "scalar"} : (i64) -> !llvm.ptr
    %15 = llvm.mlir.zero : !llvm.ptr
    %16 = llvm.mlir.constant(27 : i32) : i32
    %17 = llvm.getelementptr %15[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %20 = llvm.insertvalue %18, %19[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %21 = llvm.mlir.constant(20180515 : i32) : i32
    %22 = llvm.insertvalue %21, %20[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.trunc %23 : i32 to i8
    %25 = llvm.insertvalue %24, %22[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %26 = llvm.trunc %16 : i32 to i8
    %27 = llvm.insertvalue %26, %25[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %28 = llvm.mlir.constant(2 : i32) : i32
    %29 = llvm.trunc %28 : i32 to i8
    %30 = llvm.insertvalue %29, %27[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %31 = llvm.insertvalue %24, %30[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %32 = llvm.insertvalue %15, %31[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    llvm.store %32, %6 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
    %33 = llvm.load %6 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    llvm.store %33, %14 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
    %34 = llvm.mlir.addressof @_QFEsect_arr : !llvm.ptr
    %35 = llvm.mul %18, %7  : i64
    %36 = llvm.call @malloc(%35) {operandSegmentSizes = array<i32: 0, 1>, uniq_name = "_QFEfull_arr.alloc"} : (i64) -> !llvm.ptr
    %37 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %38 = llvm.insertvalue %18, %37[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %39 = llvm.insertvalue %21, %38[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %40 = llvm.trunc %0 : i32 to i8
    %41 = llvm.insertvalue %40, %39[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %42 = llvm.insertvalue %26, %41[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %43 = llvm.insertvalue %29, %42[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %44 = llvm.insertvalue %24, %43[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %45 = llvm.insertvalue %13, %44[7, 0, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %46 = llvm.insertvalue %7, %45[7, 0, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %47 = llvm.insertvalue %18, %46[7, 0, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %48 = llvm.insertvalue %36, %47[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    llvm.store %48, %5 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    %49 = llvm.load %5 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.store %49, %12 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    %50 = llvm.getelementptr %15[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.mul %51, %7  : i64
    %53 = llvm.call @malloc(%52) {operandSegmentSizes = array<i32: 0, 1>, uniq_name = "_QFEsect_arr.alloc"} : (i64) -> !llvm.ptr
    %54 = llvm.mlir.constant(9 : i32) : i32
    %55 = llvm.insertvalue %51, %37[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %56 = llvm.insertvalue %21, %55[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %57 = llvm.insertvalue %40, %56[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %58 = llvm.trunc %54 : i32 to i8
    %59 = llvm.insertvalue %58, %57[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %60 = llvm.insertvalue %29, %59[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %61 = llvm.insertvalue %24, %60[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %62 = llvm.insertvalue %13, %61[7, 0, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %63 = llvm.insertvalue %7, %62[7, 0, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %64 = llvm.insertvalue %51, %63[7, 0, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %65 = llvm.insertvalue %53, %64[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    llvm.store %65, %4 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    %66 = llvm.load %4 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.store %66, %34 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    %67 = llvm.call @malloc(%18) {operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFEscalar.alloc"} : (i64) -> !llvm.ptr
    %68 = llvm.insertvalue %67, %31[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    llvm.store %68, %3 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
    %69 = llvm.load %3 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    llvm.store %69, %14 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
    %70 = llvm.load %12 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.store %70, %2 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    %71 = llvm.getelementptr %2[0, 7, %11, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %72 = llvm.load %71 : !llvm.ptr -> i64
    %73 = llvm.getelementptr %2[0, 7, %11, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %74 = llvm.load %73 : !llvm.ptr -> i64
    %75 = llvm.getelementptr %2[0, 7, %11, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %76 = llvm.load %75 : !llvm.ptr -> i64
    %77 = llvm.sub %74, %10  : i64
    %78 = omp.bounds lower_bound(%11 : i64) upper_bound(%77 : i64) stride(%76 : i64) start_idx(%72 : i64) {stride_in_bytes = true}
    %79 = omp.map_info var_ptr(%12 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) bounds(%78) -> !llvm.ptr {is_fortran_allocatable, name = "full_arr"}
    %80 = llvm.load %34 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.store %80, %1 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    %81 = llvm.getelementptr %1[0, 7, %11, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %82 = llvm.load %81 : !llvm.ptr -> i64
    %83 = llvm.getelementptr %1[0, 7, %11, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %84 = llvm.getelementptr %1[0, 7, %11, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %85 = llvm.load %84 : !llvm.ptr -> i64
    %86 = llvm.sub %9, %82  : i64
    %87 = llvm.sub %8, %82  : i64
    %88 = omp.bounds lower_bound(%86 : i64) upper_bound(%87 : i64) stride(%85 : i64) start_idx(%82 : i64) {stride_in_bytes = true}
    %89 = omp.map_info var_ptr(%34 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) bounds(%88) -> !llvm.ptr {is_fortran_allocatable, name = "sect_arr(2:5)"}
    %90 = omp.map_info var_ptr(%14 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {is_fortran_allocatable, name = "scalar"}
    omp.target map_entries(%79 -> %arg0, %89 -> %arg1, %90 -> %arg2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
    ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr):
      %91 = llvm.mlir.constant(1 : i32) : i32
      %92 = llvm.alloca %91 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
      %93 = llvm.alloca %91 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
      %94 = llvm.alloca %91 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
      %95 = llvm.alloca %91 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
      %96 = llvm.mlir.constant(true) : i1
      %97 = llvm.mlir.constant(false) : i1
      %98 = llvm.mlir.constant(0 : i64) : i64
      %99 = llvm.mlir.constant(3 : i64) : i64
      %100 = llvm.mlir.constant(1 : i64) : i64
      %101 = llvm.mlir.constant(0 : index) : i64
      %102 = llvm.mlir.constant(1.000000e+00 : f32) : f32
      %103 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      llvm.store %103, %95 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
      %104 = llvm.getelementptr %95[0, 7, %101, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %105 = llvm.load %104 : !llvm.ptr -> i64
      %106 = llvm.getelementptr %95[0, 7, %101, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %107 = llvm.getelementptr %95[0, 7, %101, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %108 = llvm.getelementptr %95[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %109 = llvm.load %108 : !llvm.ptr -> !llvm.ptr
      %110 = llvm.sub %100, %105  : i64
      %111 = llvm.getelementptr %109[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %102, %111 : f32, !llvm.ptr
      %112 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      llvm.store %112, %94 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
      %113 = llvm.getelementptr %94[0, 7, %101, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %114 = llvm.load %113 : !llvm.ptr -> i64
      %115 = llvm.getelementptr %94[0, 7, %101, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %116 = llvm.getelementptr %94[0, 7, %101, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %117 = llvm.getelementptr %94[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %118 = llvm.load %117 : !llvm.ptr -> !llvm.ptr
      %119 = llvm.sub %99, %114  : i64
      %120 = llvm.getelementptr %118[%119] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %91, %120 : i32, !llvm.ptr
      %121 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
      llvm.store %121, %93 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
      %122 = llvm.getelementptr %93[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
      %123 = llvm.load %122 : !llvm.ptr -> !llvm.ptr
      %124 = llvm.ptrtoint %123 : !llvm.ptr to i64
      %125 = llvm.icmp "ne" %124, %98 : i64
      llvm.cond_br %125, ^bb2(%97, %123 : i1, !llvm.ptr), ^bb1
    ^bb1:  // pred: ^bb0
      %126 = llvm.mlir.zero : !llvm.ptr
      %127 = llvm.getelementptr %126[1] : (!llvm.ptr) -> !llvm.ptr, f32
      %128 = llvm.ptrtoint %127 : !llvm.ptr to i64
      %129 = llvm.call @malloc(%128) {in_type = f32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = ".auto.alloc"} : (i64) -> !llvm.ptr
      llvm.br ^bb2(%96, %129 : i1, !llvm.ptr)
    ^bb2(%130: i1, %131: !llvm.ptr):  // 2 preds: ^bb0, ^bb1
      %132 = llvm.sitofp %91 : i32 to f32
      llvm.store %132, %131 : f32, !llvm.ptr
      llvm.cond_br %130, ^bb3, ^bb6
    ^bb3:  // pred: ^bb2
      llvm.cond_br %125, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llvm.call @free(%123) : (!llvm.ptr) -> ()
      llvm.br ^bb5
    ^bb5:  // 2 preds: ^bb3, ^bb4
      %133 = llvm.mlir.constant(27 : i32) : i32
      %134 = llvm.mlir.zero : !llvm.ptr
      %135 = llvm.getelementptr %134[1] : (!llvm.ptr) -> !llvm.ptr, f32
      %136 = llvm.ptrtoint %135 : !llvm.ptr to i64
      %137 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
      %138 = llvm.insertvalue %136, %137[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
      %139 = llvm.mlir.constant(20180515 : i32) : i32
      %140 = llvm.insertvalue %139, %138[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
      %141 = llvm.mlir.constant(0 : i32) : i32
      %142 = llvm.trunc %141 : i32 to i8
      %143 = llvm.insertvalue %142, %140[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
      %144 = llvm.trunc %133 : i32 to i8
      %145 = llvm.insertvalue %144, %143[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
      %146 = llvm.mlir.constant(2 : i32) : i32
      %147 = llvm.trunc %146 : i32 to i8
      %148 = llvm.insertvalue %147, %145[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
      %149 = llvm.insertvalue %142, %148[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
      %150 = llvm.insertvalue %131, %149[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
      llvm.store %150, %92 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
      %151 = llvm.load %92 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
      llvm.store %151, %arg2 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
      llvm.br ^bb6
    ^bb6:  // 2 preds: ^bb2, ^bb5
      omp.terminator
    }
    llvm.return
  }
  llvm.mlir.global internal @_QFEfull_arr() {addr_space = 0 : i32} : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(27 : i32) : i32
    %3 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %6 = llvm.insertvalue %4, %5[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %7 = llvm.mlir.constant(20180515 : i32) : i32
    %8 = llvm.insertvalue %7, %6[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.trunc %9 : i32 to i8
    %11 = llvm.insertvalue %10, %8[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %12 = llvm.trunc %2 : i32 to i8
    %13 = llvm.insertvalue %12, %11[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %14 = llvm.mlir.constant(2 : i32) : i32
    %15 = llvm.trunc %14 : i32 to i8
    %16 = llvm.insertvalue %15, %13[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.trunc %17 : i32 to i8
    %19 = llvm.insertvalue %18, %16[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %20 = llvm.mlir.constant(1 : i64) : i64
    %21 = llvm.insertvalue %20, %19[7, 0, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %22 = llvm.insertvalue %0, %21[7, 0, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %23 = llvm.insertvalue %4, %22[7, 0, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %24 = llvm.insertvalue %1, %23[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    llvm.return %24 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
  }
  llvm.mlir.global internal @_QFEsect_arr() {addr_space = 0 : i32} : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(9 : i32) : i32
    %3 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %6 = llvm.insertvalue %4, %5[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %7 = llvm.mlir.constant(20180515 : i32) : i32
    %8 = llvm.insertvalue %7, %6[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.trunc %9 : i32 to i8
    %11 = llvm.insertvalue %10, %8[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %12 = llvm.trunc %2 : i32 to i8
    %13 = llvm.insertvalue %12, %11[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %14 = llvm.mlir.constant(2 : i32) : i32
    %15 = llvm.trunc %14 : i32 to i8
    %16 = llvm.insertvalue %15, %13[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.trunc %17 : i32 to i8
    %19 = llvm.insertvalue %18, %16[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %20 = llvm.mlir.constant(1 : i64) : i64
    %21 = llvm.insertvalue %20, %19[7, 0, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %22 = llvm.insertvalue %0, %21[7, 0, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %23 = llvm.insertvalue %4, %22[7, 0, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %24 = llvm.insertvalue %1, %23[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    llvm.return %24 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
  }
  llvm.mlir.global linkonce constant @_QQcl.6606f061cfd55e600f1165aa97a647a5() comdat(@__llvm_comdat::@_QQcl.6606f061cfd55e600f1165aa97a647a5) {addr_space = 0 : i32} : !llvm.array<97 x i8> {
    %0 = llvm.mlir.constant("/home/agozillo/git/flang-dev/work-dir/declare-target-map/omptarget-fortran-allocatables-test.f90\00") : !llvm.array<97 x i8>
    llvm.return %0 : !llvm.array<97 x i8>
  }
  llvm.comdat @__llvm_comdat {
    llvm.comdat_selector @_QQcl.6606f061cfd55e600f1165aa97a647a5 any
  }
  llvm.mlir.global external constant @_QQEnvironmentDefaults() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}

// CHECK: @.offload_sizes = private unnamed_addr constant [9 x i64] [i64 0, i64 48, i64 0, i64 0, i64 48, i64 0, i64 0, i64 24, i64 0]
// CHECK: @.offload_maptypes = private unnamed_addr constant [9 x i64] [i64 32, i64 281474976710657, i64 281474976710675, i64 32, i64 1125899906842625, i64 1125899906842643, i64 32, i64 1970324836974593, i64 1970324836974611]
// CHECK: @.offload_mapnames = private constant [9 x ptr] [ptr @0, ptr @0, ptr @0, ptr @1, ptr @1, ptr @1, ptr @2, ptr @2, ptr @2]

// CHECK: define void @_QQmain()

// CHECK: %[[ARR_SECT_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
// CHECK: %[[FULL_ARR_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
// CHECK: %[[SCALAR_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1, align 8

// CHECK: %[[FULL_ARR_UB:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[FULL_ARR_ALLOCA]], i32 0, i32 7, i64 0, i32 1
// CHECK: %[[LFULL_ARR_UB:.*]] = load i64, ptr %[[FULL_ARR_UB]], align 4
// CHECK: %[[FULL_ARR_UB2:.*]] = sub i64 %[[LFULL_ARR_UB]], 1

// CHECK: %[[ARR_SECT_LB:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ARR_SECT_ALLOCA]], i32 0, i32 7, i64 0, i32 0
// CHECK: %[[LARR_SECT_LB:.*]] = load i64, ptr %[[ARR_SECT_LB]], align 4
// CHECK: %[[ARR_SECT_BOUNDS:.*]] = sub i64 2, %[[LARR_SECT_LB]]
// CHECK: %[[ARR_SECT_BOUNDS2:.*]] = sub i64 5, %[[LARR_SECT_LB]]

// CHECK: %[[FULL_ARR_UB3:.*]] = sub i64 %[[FULL_ARR_UB2]], 0
// CHECK: %[[FULL_ARR_UB4:.*]] = add i64 %[[FULL_ARR_UB3]], 1
// CHECK: %[[FULL_ARR_UB5:.*]] = mul i64 1, %[[FULL_ARR_UB4]]
// CHECK: %[[FULL_ARR_TYPE_SIZE:.*]] = load i64, ptr getelementptr inbounds ({ ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr @_QFEfull_arr, i64 0, i32 1), align 4
// CHECK: %[[FULL_ARR_SIZE:.*]] = mul i64 %[[FULL_ARR_UB5]], %[[FULL_ARR_TYPE_SIZE]]

// CHECK: %[[ARR_SECT_RANGE:.*]] = sub i64 %[[ARR_SECT_BOUNDS2]], %[[ARR_SECT_BOUNDS]]
// CHECK: %[[ARR_SECT_ELEMENTS:.*]] = add i64 %[[ARR_SECT_RANGE]], 1
// CHECK: %[[ARR_SECT_ELEMENTS2:.*]] = mul i64 1, %[[ARR_SECT_ELEMENTS]]
// CHECK: %[[ARR_SECT_TYPE_SIZE:.*]] = load i64, ptr getelementptr inbounds ({ ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr @_QFEsect_arr, i64 0, i32 1), align 4
// CHECK: %[[ARR_SECT_SIZE:.*]] = mul i64 %[[ARR_SECT_ELEMENTS2]], %[[ARR_SECT_TYPE_SIZE]]

// CHECK: %[[SCALAR_TYPE_SIZE:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[SCALAR_ALLOCA]], i64 0, i32 1
// CHECK: %[[SCALAR_SIZE:.*]] = load i64, ptr %[[SCALAR_TYPE_SIZE]], align 4

// CHECK: %[[FULL_ARR_DESC_SIZE:.*]] = sdiv exact i64 sub (i64 ptrtoint (ptr getelementptr inbounds ({ ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr @_QFEfull_arr, i32 1) to i64), i64 ptrtoint (ptr @_QFEfull_arr to i64)), ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[FULL_ARR_ELE_SZ:.*]] = load i64, ptr getelementptr inbounds ({ ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr @_QFEfull_arr, i64 0, i32 1), align 4
// CHECK: %[[FULL_ARR_OFFSET:.*]] = mul i64 0, %[[FULL_ARR_ELE_SZ]]
// CHECK: %[[LFULL_ARR:.*]] = load ptr, ptr @_QFEfull_arr, align 8
// CHECK: %[[FULL_ARR_PTR:.*]] = getelementptr i8, ptr %[[LFULL_ARR]], i64 %[[FULL_ARR_OFFSET]]

// CHECK: %[[ARR_SECT_DESC_SIZE:.*]] = sdiv exact i64 sub (i64 ptrtoint (ptr getelementptr inbounds ({ ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr @_QFEsect_arr, i32 1) to i64), i64 ptrtoint (ptr @_QFEsect_arr to i64)), ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[ARR_SECT_ELE_SZ:.*]] = load i64, ptr getelementptr inbounds ({ ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr @_QFEsect_arr, i64 0, i32 1), align 4
// CHECK: %[[ARR_SECT_OFFSET:.*]] = mul i64 %[[ARR_SECT_BOUNDS]], %[[ARR_SECT_ELE_SZ]]
// CHECK: %[[LARR_SECT:.*]] = load ptr, ptr @_QFEsect_arr, align 8
// CHECK: %[[ARR_SECT_PTR:.*]] = getelementptr i8, ptr %[[LARR_SECT]], i64 %[[ARR_SECT_OFFSET]]

// CHECK: %[[SCALAR_DESC_CALC1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[SCALAR_ALLOCA]], i32 1
// CHECK: %[[SCALAR_DESC_CALC2:.*]] = ptrtoint ptr %[[SCALAR_DESC_CALC1]] to i64
// CHECK: %[[SCALAR_DESC_CALC3:.*]] = ptrtoint ptr %[[SCALAR_ALLOCA]] to i64
// CHECK: %[[SCALAR_DESC_CALC4:.*]] = sub i64 %[[SCALAR_DESC_CALC2]], %[[SCALAR_DESC_CALC3]]
// CHECK: %[[SCALAR_DESC_SZ:.*]] = sdiv exact i64 %[[SCALAR_DESC_CALC4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[SCALAR_BASE:.*]] = getelementptr inbounds { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[SCALAR_ALLOCA]], i64 0, i32 0
// CHECK: %[[LSCALAR_BASE:.*]] = load ptr, ptr %[[SCALAR_BASE]], align 8
// CHECK: %[[SCALAR_PTR:.*]] = getelementptr i8, ptr %[[LSCALAR_BASE]], i64 0

// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK: store ptr @_QFEfull_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK: store ptr @_QFEfull_arr, ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK: store i64 %[[FULL_ARR_DESC_SIZE]], ptr %[[OFFLOADSIZES]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK: store ptr @_QFEfull_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK: store ptr @_QFEfull_arr, ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK: store ptr @_QFEfull_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK: store ptr %[[FULL_ARR_PTR]], ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK: store i64 %[[FULL_ARR_SIZE]], ptr %[[OFFLOADSIZES]], align 8

// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK: store ptr @_QFEsect_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK: store ptr @_QFEsect_arr, ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 3
// CHECK: store i64 %[[ARR_SECT_DESC_SIZE]], ptr %[[OFFLOADSIZES]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
// CHECK: store ptr @_QFEsect_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 4
// CHECK: store ptr @_QFEsect_arr, ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
// CHECK: store ptr @_QFEsect_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 5
// CHECK: store ptr %[[ARR_SECT_PTR]], ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 5
// CHECK: store i64 %[[ARR_SECT_SIZE]], ptr %[[OFFLOADSIZES]], align 8

// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
// CHECK: store ptr %[[SCALAR_ALLOCA]], ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 6
// CHECK: store ptr %[[SCALAR_ALLOCA]], ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 6
// CHECK: store i64 %[[SCALAR_DESC_SZ]], ptr %[[OFFLOADSIZES]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 7
// CHECK: store ptr %[[SCALAR_ALLOCA]], ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 7
// CHECK: store ptr %[[SCALAR_ALLOCA]], ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 8
// CHECK: store ptr %[[SCALAR_BASE]], ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 8
// CHECK: store ptr %[[SCALAR_PTR]], ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 8
// CHECK: store i64 %[[SCALAR_SIZE]], ptr %[[OFFLOADSIZES]], align 8
