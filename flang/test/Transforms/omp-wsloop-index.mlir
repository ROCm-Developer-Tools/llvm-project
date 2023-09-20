// RUN: fir-opt --omp-loop-index-mem2reg %s | FileCheck %s

func.func private @foo(%arg0 : !fir.ref<i32>) -> i32

// CHECK-LABEL: @remove_alloca
func.func @remove_alloca() {
  // CHECK: %[[RESULT:.*]] = fir.alloca i32
  // CHECK-NEXT: omp.parallel
  %0 = fir.alloca i32
  omp.parallel {
    // CHECK-NOT: fir.alloca
    // CHECK-NEXT: arith.constant 1
    // CHECK-NEXT: arith.constant 10
    // CHECK-NEXT: omp.wsloop for (%[[INDEX:.*]]) : i32
    %1 = fir.alloca i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    omp.wsloop for (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK-NOT: fir.alloca
      // CHECK-NEXT: fir.store %[[INDEX]] to %[[RESULT]]
      // CHECK-NEXT: omp.yield
      fir.store %arg0 to %1 : !fir.ref<i32>
      %2 = fir.load %1 : !fir.ref<i32>
      fir.store %2 to %0 : !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }// TODO Add simdloop
  return
}

// CHECK-LABEL: @push_alloca
func.func @push_alloca() {
  // CHECK: %[[RESULT:.*]] = fir.alloca i32
  // CHECK-NEXT: omp.parallel
  %0 = fir.alloca i32
  omp.parallel {
    // CHECK-NOT: fir.alloca
    // CHECK-NEXT: arith.constant 1
    // CHECK-NEXT: arith.constant 10
    // CHECK-NEXT: omp.wsloop for (%[[INDEX:.*]]) : i32
    %1 = fir.alloca i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    omp.wsloop for (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK-NEXT: %[[ALLOCA:.*]] = fir.alloca i32
      // CHECK-NEXT: fir.store %[[INDEX]] to %[[ALLOCA]]
      // CHECK-NEXT: %[[RETURN:.*]] = func.call @foo(%[[ALLOCA]])
      // CHECK-NEXT: fir.store %[[RETURN]] to %[[RESULT]]
      // CHECK-NEXT: omp.yield
      fir.store %arg0 to %1 : !fir.ref<i32>
      %2 = func.call @foo(%1) : (!fir.ref<i32>) -> i32
      fir.store %2 to %0 : !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  } // TODO Add simdloop
  return
}

// TODO Add HLFIR variants
