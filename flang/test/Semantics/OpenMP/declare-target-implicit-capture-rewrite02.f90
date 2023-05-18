! RUN: %flang_fc1 -fopenmp -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s
!
! Ensure that functions and subroutines referenced within 
! declare target functions are themselves made declare target 
! when inside of an interface as specified by more recent 
! iterations of the OpenMP specification. This is done through 
! a semantic pass which appends the implicitly captured functions 
! to the original declare target declaration rather than 
! generating and inserting new ones within the captured functions.
! 
! For example a declare target inside of a function named 'ORIGINAL', 
! would initially be empty, after the pass, the declare target 
! would be expanded to declare target to(ORIGINAL). If 
! there is a function named 'CAPTURED' called within 'ORIGINAL' 
! the declare target inside of 'ORIGINAL' would be further 
! expanded to declare target to(ORIGINAL, CAPTURED)
!
! This test case is declare-target-implicit-capture-rewrite01.f90
! except placed into a module, to verify the pass works and continues
! to work in conjunction with modules.

module test_module
    contains
    FUNCTION IMPLICITLY_CAPTURED_NEST_TWICE() RESULT(I)
        INTEGER :: I
        I = 10
    END FUNCTION IMPLICITLY_CAPTURED_NEST_TWICE
    
    FUNCTION IMPLICITLY_CAPTURED_ONE_TWICE() RESULT(K)
        K = IMPLICITLY_CAPTURED_NEST_TWICE()
    END FUNCTION IMPLICITLY_CAPTURED_ONE_TWICE
    
    FUNCTION IMPLICITLY_CAPTURED_TWO_TWICE() RESULT(Y)
        INTEGER :: Y
        Y = 5
    END FUNCTION IMPLICITLY_CAPTURED_TWO_TWICE
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'target_function_test_device'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'implicitly_captured_one_twice'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'implicitly_captured_two_twice'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'implicitly_captured_nest_twice'
    ! CHECK: OmpClause -> DeviceType -> OmpDeviceTypeClause -> Type = Nohost
    FUNCTION TARGET_FUNCTION_TEST_DEVICE() RESULT(J)
    !$omp declare target to(TARGET_FUNCTION_TEST_DEVICE) device_type(nohost)
        INTEGER :: I, J
        I = IMPLICITLY_CAPTURED_ONE_TWICE()
        J = IMPLICITLY_CAPTURED_TWO_TWICE() + I 
    END FUNCTION TARGET_FUNCTION_TEST_DEVICE
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'target_function_test_host'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'implicitly_captured_one_twice'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'implicitly_captured_two_twice'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'implicitly_captured_nest_twice'
    ! CHECK: OmpClause -> DeviceType -> OmpDeviceTypeClause -> Type = Host
    FUNCTION TARGET_FUNCTION_TEST_HOST() RESULT(J)
    !$omp declare target to(TARGET_FUNCTION_TEST_HOST) device_type(host)
        INTEGER :: I, J
        I = IMPLICITLY_CAPTURED_ONE_TWICE()
        J = IMPLICITLY_CAPTURED_TWO_TWICE() + I
    END FUNCTION TARGET_FUNCTION_TEST_HOST
    
    !! -----
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'func_t_device'
    ! CHECK: OmpClause -> DeviceType -> OmpDeviceTypeClause -> Type = Nohost
    FUNCTION FUNC_T_DEVICE() RESULT(I)
    !$omp declare target to(FUNC_T_DEVICE) device_type(nohost)
        INTEGER :: I
        I = 1
    END FUNCTION FUNC_T_DEVICE
    
    !! -----
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'subr_t_any'
    ! CHECK: OmpClause -> DeviceType -> OmpDeviceTypeClause -> Type = Any
    SUBROUTINE SUBR_T_ANY()
    !$omp declare target to(SUBR_T_ANY) device_type(any)
    END
    
    !! -----
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithList -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'subr_default_extendedlist'
    SUBROUTINE SUBR_DEFAULT_EXTENDEDLIST()
    !$omp declare target(SUBR_DEFAULT_EXTENDEDLIST)
    END
    
    !! -----
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'subr_unspecified'
    SUBROUTINE SUBR_UNSPECIFIED()
    !$omp declare target
    END
    
    !! -----
    
    FUNCTION UNSPECIFIED_CAPTURE() RESULT(K)
        REAL :: K
        K = 1
    END FUNCTION UNSPECIFIED_CAPTURE
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'subr_unspecified_capture'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'unspecified_capture'
    SUBROUTINE SUBR_UNSPECIFIED_CAPTURE()
    !$omp declare target
        REAL :: I
        I = UNSPECIFIED_CAPTURE()
    END
    
    !! -----
        
    FUNCTION IMPLICITLY_CAPTURED_NEST() RESULT(K)
        INTEGER :: I
        I = 10
        K = I
    END FUNCTION IMPLICITLY_CAPTURED_NEST
    
    FUNCTION IMPLICITLY_CAPTURED_ONE() RESULT(K)
        K = IMPLICITLY_CAPTURED_NEST()
    END FUNCTION IMPLICITLY_CAPTURED_ONE
    
    FUNCTION IMPLICITLY_CAPTURED_TWO() RESULT(K)
        INTEGER :: I
        I = 10
        K = I
    END FUNCTION IMPLICITLY_CAPTURED_TWO
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'target_function_test'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'implicitly_captured_one'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'implicitly_captured_two'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'implicitly_captured_nest'
    ! CHECK: OmpClause -> DeviceType -> OmpDeviceTypeClause -> Type = Nohost
    FUNCTION TARGET_FUNCTION_TEST() RESULT(J)
    !$omp declare target to(TARGET_FUNCTION_TEST) device_type(nohost)
        INTEGER :: I, J
        I = IMPLICITLY_CAPTURED_ONE()
        J = IMPLICITLY_CAPTURED_TWO() + I
    END FUNCTION TARGET_FUNCTION_TEST
    
    !! -----
    
    FUNCTION NO_DECLARE_TARGET() RESULT(K)
        implicit none
        REAL :: I, K
        I = 10.0
        K = I
    END FUNCTION NO_DECLARE_TARGET
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'declare_target_two'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'no_declare_target'
    FUNCTION DECLARE_TARGET_TWO() RESULT(J)
    !$omp declare target to(DECLARE_TARGET_TWO)
        implicit none
        REAL :: I, J
        I = NO_DECLARE_TARGET()
        J = I
    END FUNCTION DECLARE_TARGET_TWO
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'declare_target_one'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'declare_target_two'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'no_declare_target'
    FUNCTION DECLARE_TARGET_ONE() RESULT(I)
    !$omp declare target to(DECLARE_TARGET_ONE)
        implicit none
        REAL :: K, I
        I = DECLARE_TARGET_TWO()
        K = I
    END FUNCTION DECLARE_TARGET_ONE
    
    !! -----
    
    RECURSIVE FUNCTION IMPLICITLY_CAPTURED_RECURSIVE(INCREMENT) RESULT(K)
        INTEGER :: INCREMENT, K
        IF (INCREMENT == 10) THEN
            K = INCREMENT
        ELSE
            K = IMPLICITLY_CAPTURED_RECURSIVE(INCREMENT + 1)
        END IF
    END FUNCTION IMPLICITLY_CAPTURED_RECURSIVE
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithList -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'target_function_recurse'
    ! CHECK: OmpObject -> Designator -> DataRef -> Name = 'implicitly_captured_recursive'
    FUNCTION TARGET_FUNCTION_RECURSE() RESULT(I)
    !$omp declare target(TARGET_FUNCTION_RECURSE)
        INTEGER :: I
        I = IMPLICITLY_CAPTURED_RECURSIVE(0)
    END FUNCTION TARGET_FUNCTION_RECURSE
    
    !! -----
    
    ! CHECK: SpecificationPart
    ! CHECK: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
    ! CHECK: Verbatim
    ! CHECK: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'recursive_declare_target'
    RECURSIVE FUNCTION RECURSIVE_DECLARE_TARGET(INCREMENT) RESULT(K)
    !$omp declare target to(RECURSIVE_DECLARE_TARGET) device_type(nohost)
        INTEGER :: INCREMENT, K
        IF (INCREMENT == 10) THEN
            K = INCREMENT
        ELSE
            K = RECURSIVE_DECLARE_TARGET(INCREMENT + 1)
        END IF
    END FUNCTION RECURSIVE_DECLARE_TARGET    
end module test_module
