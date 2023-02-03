! Ensure argument -internal-isystem works as expected with an included header.

! RUN: not %flang_fc1 -E %s 2>&1 | FileCheck %s --check-prefix=UNINCLUDED
! RUN: %flang_fc1 -E -internal-isystem %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=INCLUDED

! UNINCLUDED:#include: Source file 'basic-header-one.h' was not found
! UNINCLUDED-NOT:program B
! UNINCLUDED-NOT:program C

! INCLUDED:program MainDirectoryOne
! INCLUDED-NOT:program X
! INCLUDED-NOT:program B
! INCLUDED:program MainDirectoryTwo
! INCLUDED-NOT:program Y
! INCLUDED-NOT:program C

! include-test-one.f90
#include <basic-header-one.h>
#ifdef X
program X
#else
program B
#endif
end

! include-test-two.f90
INCLUDE "basic-header-two.h"
#ifdef Y
program Y
#else
program C
#endif
end
