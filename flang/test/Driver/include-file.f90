! Ensure argument -include works as expected with an included header file.

! RUN: %flang_fc1 -E -include "%S/Inputs/basic-header-one.h" %s  2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang_fc1 -E -include does-not-exist.h %s  2>&1 | FileCheck %s --check-prefix=MISSING

! INCLUDED: program MainDirectoryOne
! INCLUDED-NOT: program X

! MISSING: error: Source file 'does-not-exist.h' was not found
! MISSING-NOT: program

program X
end
