add_mlir_dialect_library(MLIROpenMPDialect
  OpenMPDialect.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  MLIROpenMPOpsIncGen
  MLIROpenMPOpsInterfacesIncGen
  MLIROpenMPTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRFuncDialect
  )
