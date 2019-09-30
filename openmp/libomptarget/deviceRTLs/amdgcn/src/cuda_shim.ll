target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
target triple = "amdgcn-amd-amdhsa"

declare i64 @__ockl_get_local_size(i32) #0
declare i64 @__ockl_get_num_groups(i32) #0
declare i32 @llvm.amdgcn.workitem.id.x() #1
declare i32 @llvm.amdgcn.workgroup.id.x() #1
declare i32 @llvm.amdgcn.workitem.id.y() #1
declare i32 @llvm.amdgcn.workgroup.id.y() #1
declare i32 @llvm.amdgcn.workitem.id.z() #1
declare i32 @llvm.amdgcn.workgroup.id.z() #1

define void @__nvvm_membar_gl() #2 {
  fence syncscope("agent") seq_cst
  ret void
}

define void @__nvvm_membar_cta() #2 {
  fence syncscope("workgroup") seq_cst
  ret void
}

define void @__nvvm_membar_sys() #2 {
  fence seq_cst
  ret void
}

define i32 @__nvvm_read_ptx_sreg_tid_x() #2 {
  %id.i = tail call i32 @llvm.amdgcn.workitem.id.x() #3
  ret i32 %id.i
}

define i32 @__nvvm_read_ptx_sreg_tid_y() #2 {
  %id.i = tail call i32 @llvm.amdgcn.workitem.id.y() #3
  ret i32 %id.i
}

define i32 @__nvvm_read_ptx_sreg_tid_z() #2 {
  %id.i = tail call i32 @llvm.amdgcn.workitem.id.z() #3
  ret i32 %id.i
}

define i32 @__nvvm_read_ptx_sreg_ctaid_x() #2 {
  %id.i = tail call i32 @llvm.amdgcn.workgroup.id.x() #3
  ret i32 %id.i
}

define i32 @__nvvm_read_ptx_sreg_ctaid_y() #2 {
  %id.i = tail call i32 @llvm.amdgcn.workgroup.id.y() #3
  ret i32 %id.i
}

define i32 @__nvvm_read_ptx_sreg_ctaid_z() #2 {
  %id.i = tail call i32 @llvm.amdgcn.workgroup.id.z() #3
  ret i32 %id.i
}

define i32 @__nvvm_read_ptx_sreg_ntid_x() #2 {
  %call.i = tail call i64 @__ockl_get_local_size(i32 0) #0
  %1 = trunc i64 %call.i to i32
  ret i32 %1
}

define i32 @__nvvm_read_ptx_sreg_ntid_y() #2 {
  %call.i = tail call i64 @__ockl_get_local_size(i32 1) #0
  %1 = trunc i64 %call.i to i32
  ret i32 %1
}

define i32 @__nvvm_read_ptx_sreg_ntid_z() #2 {
  %call.i = tail call i64 @__ockl_get_local_size(i32 2) #0
  %1 = trunc i64 %call.i to i32
  ret i32 %1
}

define i32 @__nvvm_read_ptx_sreg_nctaid_x() #2 {
  %call.i = tail call i64 @__ockl_get_num_groups(i32 0) #0
  %1 = trunc i64 %call.i to i32
  ret i32 %1
}

define i32 @__nvvm_read_ptx_sreg_nctaid_y() #2 {
  %call.i = tail call i64 @__ockl_get_num_groups(i32 1) #0
  %1 = trunc i64 %call.i to i32
  ret i32 %1
}

define i32 @__nvvm_read_ptx_sreg_nctaid_z() #2 {
  %call.i = tail call i64 @__ockl_get_num_groups(i32 2) #0
  %1 = trunc i64 %call.i to i32
  ret i32 %1
}

attributes #0 = { alwaysinline nounwind readnone }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { alwaysinline }
attributes #3 = { nounwind }
