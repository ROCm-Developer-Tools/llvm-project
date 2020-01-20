#ifndef AMDGCN_ACCESS_DIMENSIONS_H_INCLUDED
#define AMDGCN_ACCESS_DIMENSIONS_H_INCLUDED

#include <stdint.h>

#include "target_impl.h"

DEVICE uint32_t __amdgcn_grid_dim_x();
DEVICE uint32_t __amdgcn_grid_dim_y();
DEVICE uint32_t __amdgcn_grid_dim_z();
DEVICE uint32_t __amdgcn_block_dim_x();
DEVICE uint32_t __amdgcn_block_dim_y();
DEVICE uint32_t __amdgcn_block_dim_z();

#endif
