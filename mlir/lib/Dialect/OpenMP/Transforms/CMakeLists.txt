add_mlir_dialect_library(MLIROpenMPTransforms
  TargetOpMapCapture.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP/Transforms

  DEPENDS
  MLIROpenMPTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIROpenMPDialect
)
