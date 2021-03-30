/* Determine the wordsize from the preprocessor defines.  */

/*
  bits/wordsize.h needs an overlay header since
  default host settings do not match the offload arch.
  The logic in the ppc header for wordsize.h says if not
  ppc64 then default to 32 bit wordsize.

*/

#if defined (__AMDGCN__) || defined (__NVPTX__)
  #define __WORDSIZE  64
  #define __WORDSIZE_TIME64_COMPAT32 1
#else
  #include_next <bits/wordsize.h>
#endif
