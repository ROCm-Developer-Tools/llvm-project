! Offloading test checking interaction of a
! non-array allocatable with a target region
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    integer, allocatable :: test
    allocate(test)
    test = 10

!$omp target map(tofrom:test)
    test = 50
!$omp end target

    print *, test

    deallocate(test)
end program

! CHECK: 50
