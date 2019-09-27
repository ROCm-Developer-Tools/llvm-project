target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
target triple = "amdgcn-amd-amdhsa"

declare void @nvvm.membar.gl()
define void @__nvvm_membar_gl() #0 {
  tail call void @nvvm.membar.gl()
  ret void
}

declare void @nvvm.membar.cta()
define void @__nvvm_membar_cta() #0 {
  tail call void @nvvm.membar.cta()
  ret void
}

declare void @nvvm.membar.sys()
define void @__nvvm_membar_sys() #0 {
  tail call void @nvvm.membar.sys()
  ret void
}

declare i32 @nvvm.read.ptx.sreg.tid.x()
define i32 @__nvvm_read_ptx_sreg_tid_x() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.tid.x()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.tid.y()
define i32 @__nvvm_read_ptx_sreg_tid_y() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.tid.y()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.tid.z()
define i32 @__nvvm_read_ptx_sreg_tid_z() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.tid.z()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.ctaid.x()
define i32 @__nvvm_read_ptx_sreg_ctaid_x() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.ctaid.x()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.ctaid.y()
define i32 @__nvvm_read_ptx_sreg_ctaid_y() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.ctaid.y()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.ctaid.z()
define i32 @__nvvm_read_ptx_sreg_ctaid_z() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.ctaid.z()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.ntid.x()
define i32 @__nvvm_read_ptx_sreg_ntid_x() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.ntid.x()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.ntid.y()
define i32 @__nvvm_read_ptx_sreg_ntid_y() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.ntid.y()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.ntid.z()
define i32 @__nvvm_read_ptx_sreg_ntid_z() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.ntid.z()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.nctaid.x()
define i32 @__nvvm_read_ptx_sreg_nctaid_x() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.nctaid.x()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.nctaid.y()
define i32 @__nvvm_read_ptx_sreg_nctaid_y() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.nctaid.y()
  ret i32 %1
}

declare i32 @nvvm.read.ptx.sreg.nctaid.z()
define i32 @__nvvm_read_ptx_sreg_nctaid_z() #0 {
  %1 = tail call i32 @nvvm.read.ptx.sreg.nctaid.z()
  ret i32 %1
}

attributes #0 = {alwaysinline}
