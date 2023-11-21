module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<8>, dense<128> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<6>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<7>, dense<[160, 256, 256, 32]> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<3>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<4>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<5>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<1>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<2>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 32 : i32>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.flags = #omp.flags<openmp_device_version = 11>, omp.is_gpu = true, omp.is_target_device = true, omp.target = #omp.target<target_cpu = "generic-hsa", target_features = "">, omp.version = #omp.version<version = 11>} {
  func.func @_QPtarget_function() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
    return
  }
  func.func @_QPwrite_index_omp_outline_0(%arg0: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QPwrite_index"} {
    %0 = omp.map_entry var_ptr(%arg0 : !fir.ref<i32>)   -> !fir.ref<i32> {map_capture_type = #omp<variable_capture_kind(ByRef)>, map_type = 34 : i64, name = "new_len"}
    omp.target   map(%0 : !fir.ref<i32>) {
      %c10_i32 = arith.constant 10 : i32
      fir.store %c10_i32 to %arg0 : !fir.ref<i32>
      omp.terminator
    }
    return
  }
  func.func @_QPwrite_index_omp_outline_1(%arg0: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QPwrite_index"} {
    %0 = omp.map_entry var_ptr(%arg0 : !fir.ref<i32>)   -> !fir.ref<i32> {map_capture_type = #omp<variable_capture_kind(ByRef)>, map_type = 34 : i64, name = "new_len"}
    omp.target   map(%0 : !fir.ref<i32>) {
      %c20_i32 = arith.constant 20 : i32
      fir.store %c20_i32 to %arg0 : !fir.ref<i32>
      omp.terminator
    }
    return
  }
}
