// This test checks if OpenMPIRBuilder generates the same number of omp offload
// info nodes as Clang does. The wrong number of metadata nodes can provide
// miscompilation of the device code for enabled OpenMPIRBuilder
// RUN: %clang_cc1 -triple x86_64--unknown-linux-gnu -emit-llvm -fopenmp -fopenmp-enable-irbuilder -fopenmp-targets=amdgcn-amd-amdhsa -faddrsig  %s -o - | FileCheck --check-prefix BUILDER %s
// RUN: %clang_cc1 -triple x86_64--unknown-linux-gnu -emit-llvm -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -faddrsig  %s -o - | FileCheck --check-prefix NOBUILDER %s

void omp_offload_metadata_irbuilder_test() {
int a[256];
#pragma omp target parallel for
  for (int i = 0; i < 256; i++) {
    a[i] = i;
  }
}

//BUILDER: !omp_offload.info = !{!{{[0-9]+}}}
//NOBUILDER: !omp_offload.info = !{!{{[0-9]+}}}
