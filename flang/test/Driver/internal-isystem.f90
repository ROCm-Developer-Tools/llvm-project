! Ensure argument -internal-isystem works as expected with an included header.

! RUN: %flang_fc1 -E -internal-isystem %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLE
! RUN: %flang_fc1 -E -internal-isystem %S/Inputs -I %S/Inputs/header-dir %s  2>&1 | FileCheck %s --check-prefix=BOTH
! RUN: %flang_fc1 -E -I %S/Inputs/header-dir -I %S/Inputs -internal-isystem %S/Inputs/header-dir %s  2>&1 | FileCheck %s --check-prefix=OVERRIDE

! System include path is scanned
! SINGLE: program MainDirectoryOne
! SINGLE-NOT: program X

! User include path takes precedence over system include path
! BOTH: program SubDirectoryOne
! BOTH-NOT: program MainDirectoryOne

! If a directory is both specified as user and system path, process as system
! directory. Then, it will take less priority than other user paths.
! OVERRIDE: program MainDirectoryOne
! OVERRIDE-NOT: program SubDirectoryOne

#include <basic-header-one.h>
program X
end
