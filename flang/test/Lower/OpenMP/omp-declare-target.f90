!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

! Check specification valid forms of declare target with functions 
! utilising device_type and to clauses as well as the default 
! zero clause declare target

! CHECK-LABEL: func.func @_QPfunc_t_device()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(nohost)>{{.*}}
FUNCTION FUNC_T_DEVICE() RESULT(I)
!$omp declare target to(FUNC_T_DEVICE) device_type(nohost)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_T_DEVICE

! CHECK-LABEL: func.func @_QPfunc_t_host()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(host)>{{.*}}
FUNCTION FUNC_T_HOST() RESULT(I)
!$omp declare target to(FUNC_T_HOST) device_type(host)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_T_HOST

! CHECK-LABEL: func.func @_QPfunc_t_any()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
FUNCTION FUNC_T_ANY() RESULT(I)
!$omp declare target to(FUNC_T_ANY) device_type(any)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_T_ANY
    
! CHECK-LABEL: func.func @_QPfunc_default_t_any()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
FUNCTION FUNC_DEFAULT_T_ANY() RESULT(I)
!$omp declare target to(FUNC_DEFAULT_T_ANY)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_DEFAULT_T_ANY

! CHECK-LABEL: func.func @_QPfunc_default_any()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
FUNCTION FUNC_DEFAULT_ANY() RESULT(I)
!$omp declare target
    INTEGER :: I
    I = 1
END FUNCTION FUNC_DEFAULT_ANY

! CHECK-LABEL: func.func @_QPfunc_default_extendedlist()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
FUNCTION FUNC_DEFAULT_EXTENDEDLIST() RESULT(I)
!$omp declare target(FUNC_DEFAULT_EXTENDEDLIST)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_DEFAULT_EXTENDEDLIST

! CHECK-LABEL: func.func @_QPexist_on_both()
! CHECK-NOT: {{.*}}attributes {omp.declare_target = #omp<device_type({{.*}})>{{.*}}
FUNCTION EXIST_ON_BOTH() RESULT(I)
    INTEGER :: I
    I = 1
END FUNCTION EXIST_ON_BOTH

!! -----

! Check specification valid forms of declare target with subroutines 
! utilising device_type and to clauses as well as the default 
! zero clause declare target

! CHECK-LABEL: func.func @_QPsubr_t_device()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(nohost)>{{.*}}
SUBROUTINE SUBR_T_DEVICE()
!$omp declare target to(SUBR_T_DEVICE) device_type(nohost)
END

! CHECK-LABEL: func.func @_QPsubr_t_host()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(host)>{{.*}}
SUBROUTINE SUBR_T_HOST()
!$omp declare target to(SUBR_T_HOST) device_type(host)
END

! CHECK-LABEL: func.func @_QPsubr_t_any()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
SUBROUTINE SUBR_T_ANY()
!$omp declare target to(SUBR_T_ANY) device_type(any)
END

! CHECK-LABEL: func.func @_QPsubr_default_t_any()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
SUBROUTINE SUBR_DEFAULT_T_ANY()
!$omp declare target to(SUBR_DEFAULT_T_ANY)
END

! CHECK-LABEL: func.func @_QPsubr_default_any()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
SUBROUTINE SUBR_DEFAULT_ANY()
!$omp declare target
END

! CHECK-LABEL: func.func @_QPsubr_default_extendedlist()
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
SUBROUTINE SUBR_DEFAULT_EXTENDEDLIST()
!$omp declare target(SUBR_DEFAULT_EXTENDEDLIST)
END

! CHECK-LABEL: func.func @_QPsubr_exist_on_both()
! CHECK-NOT: {{.*}}attributes {omp.declare_target = #omp<device_type({{.*}})>{{.*}}
SUBROUTINE SUBR_EXIST_ON_BOTH()
END

!! -----

! Check declare target inconjunction with implicitly 
! invoked functions, this tests the declare target 
! implicit capture pass within Flang. Functions 
! invoked within an explicitly declare target function 
! are marked as declare target with the callers 
! device_type clause

! CHECK-LABEL: func.func @_QPimplicitly_captured
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
FUNCTION IMPLICITLY_CAPTURED(TOGGLE) RESULT(K)
    INTEGER :: I, J, K
    LOGICAL :: TOGGLE
    I = 10
    J = 5
    IF (TOGGLE) THEN
        K = I
    ELSE
        K = J
    END IF
END FUNCTION IMPLICITLY_CAPTURED


! CHECK-LABEL: func.func @_QPtarget_function
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
FUNCTION TARGET_FUNCTION(TOGGLE) RESULT(I)
!$omp declare target
    INTEGER :: I
    LOGICAL :: TOGGLE
    I = IMPLICITLY_CAPTURED(TOGGLE)
END FUNCTION TARGET_FUNCTION

!! -----

! Check declare target inconjunction with implicitly 
! invoked functions, this tests the declare target 
! implicit capture pass within Flang. Functions 
! invoked within an explicitly declare target function 
! are marked as declare target with the callers 
! device_type clause, however, if they are found with 
! distinct device_type clauses i.e. host and nohost, 
! then they should be marked as any

! CHECK-LABEL: func.func @_QPimplicitly_captured_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
FUNCTION IMPLICITLY_CAPTURED_TWICE() RESULT(K)
    INTEGER :: I
    I = 10
    K = I
END FUNCTION IMPLICITLY_CAPTURED_TWICE

! CHECK-LABEL: func.func @_QPtarget_function_twice_host
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(host)>{{.*}}
FUNCTION TARGET_FUNCTION_TWICE_HOST() RESULT(I)
!$omp declare target to(TARGET_FUNCTION_TWICE_HOST) device_type(host)
    INTEGER :: I
    I = IMPLICITLY_CAPTURED_TWICE()
END FUNCTION TARGET_FUNCTION_TWICE_HOST

! CHECK-LABEL: func.func @_QPtarget_function_twice_device
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(nohost)>{{.*}}
FUNCTION TARGET_FUNCTION_TWICE_DEVICE() RESULT(I)
!$omp declare target to(TARGET_FUNCTION_TWICE_DEVICE) device_type(nohost)
    INTEGER :: I
    I = IMPLICITLY_CAPTURED_TWICE()
END FUNCTION TARGET_FUNCTION_TWICE_DEVICE

!! -----

! Check declare target inconjunction with implicitly 
! invoked functions, this tests the declare target 
! implicit capture pass within Flang. A slightly more 
! complex test checking functions are marked implicitly
! appropriately. 

! CHECK-LABEL: func.func @_QPimplicitly_captured_nest
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(nohost)>{{.*}}
FUNCTION IMPLICITLY_CAPTURED_NEST() RESULT(K)
    INTEGER :: I
    I = 10
    K = I
END FUNCTION IMPLICITLY_CAPTURED_NEST

! CHECK-LABEL: func.func @_QPimplicitly_captured_one
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(nohost)>{{.*}}
FUNCTION IMPLICITLY_CAPTURED_ONE() RESULT(K)
    K = IMPLICITLY_CAPTURED_NEST()
END FUNCTION IMPLICITLY_CAPTURED_ONE

! CHECK-LABEL: func.func @_QPimplicitly_captured_two
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(nohost)>{{.*}}
FUNCTION IMPLICITLY_CAPTURED_TWO() RESULT(K)
    INTEGER :: I
    I = 10
    K = I
END FUNCTION IMPLICITLY_CAPTURED_TWO

! CHECK-LABEL: func.func @_QPtarget_function_test
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(nohost)>{{.*}}
FUNCTION TARGET_FUNCTION_TEST() RESULT(J)
!$omp declare target to(TARGET_FUNCTION_TEST) device_type(nohost)
    INTEGER :: I, J
    I = IMPLICITLY_CAPTURED_ONE()
    J = IMPLICITLY_CAPTURED_TWO() + I
END FUNCTION TARGET_FUNCTION_TEST

!! -----

! Check declare target inconjunction with implicitly 
! invoked functions, this tests the declare target 
! implicit capture pass within Flang. A slightly more 
! complex test checking functions are marked implicitly
! appropriately. 

! CHECK-LABEL: func.func @_QPimplicitly_captured_nest_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
FUNCTION IMPLICITLY_CAPTURED_NEST_TWICE() RESULT(K)
    INTEGER :: I
    I = 10
    K = I
END FUNCTION IMPLICITLY_CAPTURED_NEST_TWICE

! CHECK-LABEL: func.func @_QPimplicitly_captured_one_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
FUNCTION IMPLICITLY_CAPTURED_ONE_TWICE() RESULT(K)
    K = IMPLICITLY_CAPTURED_NEST_TWICE()
END FUNCTION IMPLICITLY_CAPTURED_ONE_TWICE

! CHECK-LABEL: func.func @_QPimplicitly_captured_two_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(any)>{{.*}}
FUNCTION IMPLICITLY_CAPTURED_TWO_TWICE() RESULT(K)
    INTEGER :: I
    I = 10
    K = I
END FUNCTION IMPLICITLY_CAPTURED_TWO_TWICE

! CHECK-LABEL: func.func @_QPtarget_function_test_device
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(nohost)>{{.*}}
FUNCTION TARGET_FUNCTION_TEST_DEVICE() RESULT(J)
    !$omp declare target to(TARGET_FUNCTION_TEST_DEVICE) device_type(nohost)
        INTEGER :: I, J
        I = IMPLICITLY_CAPTURED_ONE_TWICE()
        J = IMPLICITLY_CAPTURED_TWO_TWICE() + I 
END FUNCTION TARGET_FUNCTION_TEST_DEVICE

! CHECK-LABEL: func.func @_QPtarget_function_test_host
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp<device_type(host)>{{.*}}
FUNCTION TARGET_FUNCTION_TEST_HOST() RESULT(J)
    !$omp declare target to(TARGET_FUNCTION_TEST_HOST) device_type(host)
        INTEGER :: I, J
        I = IMPLICITLY_CAPTURED_ONE_TWICE()
        J = IMPLICITLY_CAPTURED_TWO_TWICE() + I
END FUNCTION TARGET_FUNCTION_TEST_HOST