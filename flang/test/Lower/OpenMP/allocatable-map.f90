!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefixes="HLFIRDIALECT"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefixes="LLVMOMPDIALECT"

!LLVMOMPDIALECT: %[[ALLOCA:.*]] = llvm.alloca {{.*}} x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {bindc_name = "point"} : (i64) -> !llvm.ptr
!LLVMOMPDIALECT: %[[MAP:.*]] = omp.map_info var_ptr(%[[ALLOCA]] : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(implicit, tofrom) capture(ByRef) -> !llvm.ptr {is_fortran_allocatable, name = "point"}
!LLVMOMPDIALECT: omp.target map_entries({{.*}}, %[[MAP]] -> {{.*}} : {{.*}}, !llvm.ptr) {

!HLFIRDIALECT: %[[POINTER:.*]]:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFpointer_routineEpoint"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!HLFIRDIALECT: %[[POINTER_MAP:.*]] = omp.map_info var_ptr(%[[POINTER]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {is_fortran_allocatable, name = "point"}
!HLFIRDIALECT: omp.target map_entries({{.*}}, %[[POINTER_MAP]] -> {{.*}} : {{.*}}, !fir.ref<!fir.box<!fir.ptr<i32>>>) {
subroutine pointer_routine()
    integer, pointer :: point 
!$omp target map(tofrom:pointer)
    point = 1
!$omp end target
end subroutine pointer_routine
