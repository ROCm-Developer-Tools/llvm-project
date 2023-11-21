module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", omp.flags = #omp.flags<openmp_device_version = 11>, omp.is_gpu = false, omp.is_target_device = true, omp.version = #omp.version<version = 11>} {
  func.func @_QPdevice_fn() -> i32 attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>} {
    %0 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFdevice_fnEx"}
    %c10_i32 = arith.constant 10 : i32
    fir.store %c10_i32 to %0 : !fir.ref<i32>
    %1 = fir.load %0 : !fir.ref<i32>
    return %1 : i32
  }
  func.func @_QPtarget_subr_omp_outline_0(%arg0: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QPtarget_subr"} {
    %0 = omp.map_entry var_ptr(%arg0 : !fir.ref<i32>)   -> !fir.ref<i32> {map_capture_type = #omp<variable_capture_kind(ByRef)>, map_type = 34 : i64, name = "x"}
    omp.target   map(%0 : !fir.ref<i32>) {
      %c10_i32 = arith.constant 10 : i32
      fir.store %c10_i32 to %arg0 : !fir.ref<i32>
      omp.terminator
    }
    return
  }
}
