! Ensure argument -include works as expected with an included header file.

! RUN: %flang_fc1 -E -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=UNINCLUDED
! RUN: %flang_fc1 -E -I %S/Inputs -include basic-header-one.h %s  2>&1 | FileCheck %s --check-prefix=INCLUDED

! UNINCLUDED-NOT: program X
! UNINCLUDED: program B

! INCLUDED: program MainDirectoryOne
! INCLUDED-NOT: program X
! INCLUDED-NOT: program B

#ifdef X
program X
#else
program B
#endif
end
