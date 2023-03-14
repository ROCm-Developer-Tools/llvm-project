// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: module attributes {omp.flags = #omp.flags<>} {
module attributes {omp.flags = #omp.flags<debug_kind = 0, assume_teams_oversubscription = false, assume_threads_oversubscription = false, assume_no_thread_state = false, assume_no_nested_parallelism = false>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 20>} {
module attributes {omp.flags = #omp.flags<debug_kind = 20, assume_teams_oversubscription = false, assume_threads_oversubscription = false, assume_no_thread_state = false, assume_no_nested_parallelism = false>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 100, assume_teams_oversubscription = true>} {
module attributes {omp.flags = #omp.flags<debug_kind = 100, assume_teams_oversubscription = true, assume_threads_oversubscription = false, assume_no_thread_state = false, assume_no_nested_parallelism = false>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 200, assume_teams_oversubscription = true, assume_threads_oversubscription = true>} {
module attributes {omp.flags = #omp.flags<debug_kind = 200, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = false, assume_no_nested_parallelism = false>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 300, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true>} {
module attributes {omp.flags = #omp.flags<debug_kind = 300, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = false>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 400, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = true>} {
module attributes {omp.flags = #omp.flags<debug_kind = 400, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = true>} {}

// CHECK: module attributes {omp.flags = #omp.flags<>} {
module attributes {omp.flags = #omp.flags<>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true>} {
module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true>} {
module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true>} {
module attributes {omp.flags = #omp.flags<assume_no_thread_state = true, assume_teams_oversubscription = true>} {}
// CHECK-LABEL: func @omp_decl_tar_host
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp<device_type(host)>} {
func.func @omp_decl_tar_host() -> () attributes {omp.declare_target = #omp<device_type(host)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_nohost
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp<device_type(nohost)>} {
func.func @omp_decl_tar_nohost() -> () attributes {omp.declare_target = #omp<device_type(nohost)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_any
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp<device_type(any)>} {
func.func @omp_decl_tar_any() -> () attributes {omp.declare_target = #omp<device_type(any)>} {
  return
}
