// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: module attributes {omp.rtlmoduleflags = #omp.rtlmoduleflags<debug_kind : 20, assume_teams_oversubscription : false, assume_threads_oversubscription : false, assume_no_thread_state : false, assume_no_nested_parallelism : false>} {
module attributes {omp.rtlmoduleflags = #omp.rtlmoduleflags<debug_kind : 20, assume_teams_oversubscription : false, assume_threads_oversubscription : false, assume_no_thread_state : false, assume_no_nested_parallelism : false>} {}

// -----

// CHECK: module attributes {omp.rtlmoduleflags = #omp.rtlmoduleflags<debug_kind : 0, assume_teams_oversubscription : true, assume_threads_oversubscription : false, assume_no_thread_state : false, assume_no_nested_parallelism : false>} {
module attributes {omp.rtlmoduleflags = #omp.rtlmoduleflags<debug_kind : 0, assume_teams_oversubscription : true, assume_threads_oversubscription : false, assume_no_thread_state : false, assume_no_nested_parallelism : false>} {}

// -----

// CHECK: module attributes {omp.rtlmoduleflags = #omp.rtlmoduleflags<debug_kind : 100, assume_teams_oversubscription : true, assume_threads_oversubscription : true, assume_no_thread_state : false, assume_no_nested_parallelism : false>} {
module attributes {omp.rtlmoduleflags = #omp.rtlmoduleflags<debug_kind : 100, assume_teams_oversubscription : true, assume_threads_oversubscription : true, assume_no_thread_state : false, assume_no_nested_parallelism : false>} {}

// -----

// CHECK: module attributes {omp.rtlmoduleflags = #omp.rtlmoduleflags<debug_kind : 200, assume_teams_oversubscription : true, assume_threads_oversubscription : true, assume_no_thread_state : true, assume_no_nested_parallelism : false>} {
module attributes {omp.rtlmoduleflags = #omp.rtlmoduleflags<debug_kind : 200, assume_teams_oversubscription : true, assume_threads_oversubscription : true, assume_no_thread_state : true, assume_no_nested_parallelism : false>} {}

// -----

// CHECK: module attributes {omp.rtlmoduleflags = #omp.rtlmoduleflags<debug_kind : 300, assume_teams_oversubscription : true, assume_threads_oversubscription : true, assume_no_thread_state : true, assume_no_nested_parallelism : true>} {
module attributes {omp.rtlmoduleflags = #omp.rtlmoduleflags<debug_kind : 300, assume_teams_oversubscription : true, assume_threads_oversubscription : true, assume_no_thread_state : true, assume_no_nested_parallelism : true>} {}