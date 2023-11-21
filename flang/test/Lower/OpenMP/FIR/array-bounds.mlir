module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false, omp.requires = #omp<clause_requires none>, omp.version = #omp.version<version = 11>} {
  func.func @_QPread_write_section() {
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFread_write_sectionEi"}
    %1 = fir.address_of(@_QFread_write_sectionEsp_read) : !fir.ref<!fir.array<10xi32>>
    %2 = fir.address_of(@_QFread_write_sectionEsp_write) : !fir.ref<!fir.array<10xi32>>
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %3 = omp.bounds   lower_bound(%c1_0 : index) upper_bound(%c4 : index) stride(%c1 : index) start_idx(%c1 : index)
    %4 = omp.map_info var_ptr(%1 : !fir.ref<!fir.array<10xi32>>)   map_clauses(tofrom) capture(ByRef) bounds(%3) -> !fir.ref<!fir.array<10xi32>> {name = "sp_read(2:5)"}
    %c1_1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    %c4_3 = arith.constant 4 : index
    %5 = omp.bounds   lower_bound(%c1_2 : index) upper_bound(%c4_3 : index) stride(%c1_1 : index) start_idx(%c1_1 : index)
    %6 = omp.map_info var_ptr(%2 : !fir.ref<!fir.array<10xi32>>)   map_clauses(tofrom) capture(ByRef) bounds(%5) -> !fir.ref<!fir.array<10xi32>> {name = "sp_write(2:5)"}
    %7 = omp.map_info var_ptr(%0 : !fir.ref<i32>)   map_clauses(exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {implicit = true, name = "i"}
    omp.target   map_entries(%4, %6, %7 : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>, !fir.ref<i32>) {
      %c2_i32 = arith.constant 2 : i32
      %8 = fir.convert %c2_i32 : (i32) -> index
      %c5_i32 = arith.constant 5 : i32
      %9 = fir.convert %c5_i32 : (i32) -> index
      %c1_4 = arith.constant 1 : index
      %10 = fir.convert %8 : (index) -> i32
      %11:2 = fir.do_loop %arg0 = %8 to %9 step %c1_4 iter_args(%arg1 = %10) -> (index, i32) {
        fir.store %arg1 to %0 : !fir.ref<i32>
        %12 = fir.load %0 : !fir.ref<i32>
        %13 = fir.convert %12 : (i32) -> i64
        %c1_i64 = arith.constant 1 : i64
        %14 = arith.subi %13, %c1_i64 : i64
        %15 = fir.coordinate_of %1, %14 : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
        %16 = fir.load %15 : !fir.ref<i32>
        %17 = fir.load %0 : !fir.ref<i32>
        %18 = fir.convert %17 : (i32) -> i64
        %c1_i64_5 = arith.constant 1 : i64
        %19 = arith.subi %18, %c1_i64_5 : i64
        %20 = fir.coordinate_of %2, %19 : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
        fir.store %16 to %20 : !fir.ref<i32>
        %21 = arith.addi %arg0, %c1_4 : index
        %22 = fir.convert %c1_4 : (index) -> i32
        %23 = fir.load %0 : !fir.ref<i32>
        %24 = arith.addi %23, %22 : i32
        fir.result %21, %24 : index, i32
      }
      fir.store %11#1 to %0 : !fir.ref<i32>
      omp.terminator
    }
    return
  }
  func.func @_QMassumed_array_routinesPassumed_shape_array(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr_read_write"}) {
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMassumed_array_routinesFassumed_shape_arrayEi"}
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %1:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
    %c1_0 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %2 = omp.bounds   lower_bound(%c1_0 : index) upper_bound(%c4 : index) stride(%1#2 : index) start_idx(%c1 : index) {stride_in_bytes = true}
    %3 = fir.box_addr %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    %4 = omp.map_info var_ptr(%3 : !fir.ref<!fir.array<?xi32>>)   map_clauses(tofrom) capture(ByRef) bounds(%2) -> !fir.ref<!fir.array<?xi32>> {name = "arr_read_write(2:5)"}
    %5 = omp.map_info var_ptr(%0 : !fir.ref<i32>)   map_clauses(exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {implicit = true, name = "i"}
    %6 = fir.box_addr %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    %7 = omp.map_info var_ptr(%6 : !fir.ref<!fir.array<?xi32>>)   map_clauses(exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<!fir.array<?xi32>> {implicit = true, name = ""}
    omp.target   map_entries(%4, %5, %7 : !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>, !fir.ref<!fir.array<?xi32>>) {
      %c2_i32 = arith.constant 2 : i32
      %8 = fir.convert %c2_i32 : (i32) -> index
      %c5_i32 = arith.constant 5 : i32
      %9 = fir.convert %c5_i32 : (i32) -> index
      %c1_1 = arith.constant 1 : index
      %10 = fir.convert %8 : (index) -> i32
      %11:2 = fir.do_loop %arg1 = %8 to %9 step %c1_1 iter_args(%arg2 = %10) -> (index, i32) {
        fir.store %arg2 to %0 : !fir.ref<i32>
        %12 = fir.load %0 : !fir.ref<i32>
        %13 = fir.load %0 : !fir.ref<i32>
        %14 = fir.convert %13 : (i32) -> i64
        %c1_i64 = arith.constant 1 : i64
        %15 = arith.subi %14, %c1_i64 : i64
        %16 = fir.coordinate_of %arg0, %15 : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
        fir.store %12 to %16 : !fir.ref<i32>
        %17 = arith.addi %arg1, %c1_1 : index
        %18 = fir.convert %c1_1 : (index) -> i32
        %19 = fir.load %0 : !fir.ref<i32>
        %20 = arith.addi %19, %18 : i32
        fir.result %17, %20 : index, i32
      }
      fir.store %11#1 to %0 : !fir.ref<i32>
      omp.terminator
    }
    return
  }
  func.func @_QMassumed_array_routinesPassumed_size_array(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr_read_write"}) {
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMassumed_array_routinesFassumed_size_arrayEi"}
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %1 = omp.bounds   lower_bound(%c1_0 : index) upper_bound(%c4 : index) stride(%c1 : index) start_idx(%c1 : index)
    %2 = omp.map_info var_ptr(%arg0 : !fir.ref<!fir.array<?xi32>>)   map_clauses(tofrom) capture(ByRef) bounds(%1) -> !fir.ref<!fir.array<?xi32>> {name = "arr_read_write(2:5)"}
    %3 = omp.map_info var_ptr(%0 : !fir.ref<i32>)   map_clauses(exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {implicit = true, name = "i"}
    omp.target   map_entries(%2, %3 : !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) {
      %c2_i32 = arith.constant 2 : i32
      %4 = fir.convert %c2_i32 : (i32) -> index
      %c5_i32 = arith.constant 5 : i32
      %5 = fir.convert %c5_i32 : (i32) -> index
      %c1_1 = arith.constant 1 : index
      %6 = fir.convert %4 : (index) -> i32
      %7:2 = fir.do_loop %arg1 = %4 to %5 step %c1_1 iter_args(%arg2 = %6) -> (index, i32) {
        fir.store %arg2 to %0 : !fir.ref<i32>
        %8 = fir.load %0 : !fir.ref<i32>
        %9 = fir.load %0 : !fir.ref<i32>
        %10 = fir.convert %9 : (i32) -> i64
        %c1_i64 = arith.constant 1 : i64
        %11 = arith.subi %10, %c1_i64 : i64
        %12 = fir.coordinate_of %arg0, %11 : (!fir.ref<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
        fir.store %8 to %12 : !fir.ref<i32>
        %13 = arith.addi %arg1, %c1_1 : index
        %14 = fir.convert %c1_1 : (index) -> i32
        %15 = fir.load %0 : !fir.ref<i32>
        %16 = arith.addi %15, %14 : i32
        fir.result %13, %16 : index, i32
      }
      fir.store %7#1 to %0 : !fir.ref<i32>
      omp.terminator
    }
    return
  }
  func.func @_QPcall_assumed_shape_and_size_array() {
    %c20 = arith.constant 20 : index
    %0 = fir.alloca !fir.array<20xi32> {bindc_name = "arr_read_write", uniq_name = "_QFcall_assumed_shape_and_size_arrayEarr_read_write"}
    %c1_i64 = arith.constant 1 : i64
    %1 = fir.convert %c1_i64 : (i64) -> index
    %c1_i64_0 = arith.constant 1 : i64
    %2 = fir.convert %c1_i64_0 : (i64) -> index
    %c10_i64 = arith.constant 10 : i64
    %3 = fir.convert %c10_i64 : (i64) -> index
    %4 = fir.shape %c20 : (index) -> !fir.shape<1>
    %5 = fir.slice %1, %3, %2 : (index, index, index) -> !fir.slice<1>
    %6 = fir.embox %0(%4) [%5] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
    %7 = fir.convert %6 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
    fir.call @_QMassumed_array_routinesPassumed_shape_array(%7) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
    %c10_i64_1 = arith.constant 10 : i64
    %8 = fir.convert %c10_i64_1 : (i64) -> index
    %c1_i64_2 = arith.constant 1 : i64
    %9 = fir.convert %c1_i64_2 : (i64) -> index
    %c20_i64 = arith.constant 20 : i64
    %10 = fir.convert %c20_i64 : (i64) -> index
    %11 = fir.shape %c20 : (index) -> !fir.shape<1>
    %12 = fir.slice %8, %10, %9 : (index, index, index) -> !fir.slice<1>
    %13 = fir.embox %0(%11) [%12] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<11xi32>>
    %14 = fir.box_addr %13 : (!fir.box<!fir.array<11xi32>>) -> !fir.ref<!fir.array<11xi32>>
    %15 = fir.convert %14 : (!fir.ref<!fir.array<11xi32>>) -> !fir.ref<!fir.array<?xi32>>
    fir.call @_QMassumed_array_routinesPassumed_size_array(%15) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>) -> ()
    return
  }
  fir.global internal @_QFread_write_sectionEsp_read(dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi32>) : !fir.array<10xi32>
  fir.global internal @_QFread_write_sectionEsp_write(dense<0> : tensor<10xi32>) : !fir.array<10xi32>
}
