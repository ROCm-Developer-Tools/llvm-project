

/*
  gnu/stubs.h needs an overlay header since
  default host settings do not match the offload arch.
  The logic in the ppc header for stubs.h says if not
  ppc64 then use default 32 bit stubs.
*/

#if !defined (__AMDGCN__) && !defined (__NVPTX__)
  #include_next <gnu/stubs.h>
#endif

