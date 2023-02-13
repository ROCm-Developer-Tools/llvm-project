!  Test that --offload-arch produces the expected commands

! RUN: %flang -### -c -fopenmp --offload-arch=gfx900 %s -o %t.o 2>&1 | FileCheck %s

! Host compilation
! CHECK: "-fc1"{{.*}}"-triple" "x86_64-unknown-linux-gnu"
! CHECK-SAME: "-o" "[[HOSTOUT:[^"]+]]"

! Device compilation
! CHECK-NEXT: "-fc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"
! CHECK-SAME: "-internal-isystem" "{{.*}}openmp_wrappers"
! CHECK-SAME: "-include" "__flang_openmp_device_functions.f90"
! CHECK-SAME: "-o" "[[DEVICEOUT:[^"]+]]"

! Package device bitcode
! CHECK-NEXT: clang-offload-packager
! CHECK-SAME: "-o" "[[PKGOUT:[^"]+]]"
! CHECK-SAME: "--image=file=[[DEVICEOUT]],triple=amdgcn-amd-amdhsa,arch=gfx900,kind=openmp"

! Object generation
! CHECK-NEXT: "-fc1"{{.*}}"-triple" "x86_64-unknown-linux-gnu"
! CHECK-SAME: "-fembed-offload-object=[[PKGOUT]]"
! CHECK-SAME: "[[HOSTOUT]]"
