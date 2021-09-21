//===----------- device.h - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for OpenMP Tool callback dispatchers
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_CALLBACK_H
#define _OMPTARGET_CALLBACK_H

#if (__PPC64__ | __arm__)
#define OMPT_GET_FRAME_ADDRESS(level) __builtin_frame_address(level)
#define OMPT_FRAME_POSITION_DEFAULT ompt_frame_cfa 
#else
#define OMPT_GET_FRAME_ADDRESS(level) __builtin_frame_address(level)
#define OMPT_FRAME_POSITION_DEFAULT ompt_frame_framepointer 
#endif

#define OMPT_FRAME_FLAGS (ompt_frame_runtime | OMPT_FRAME_POSITION_DEFAULT)

#define OMPT_GET_RETURN_ADDRESS(level) __builtin_return_address(level)

#include <omp-tools.h>

class OmptInterface {
 public:
  OmptInterface() : _enter_frame(NULL), _codeptr_ra(NULL), _state(ompt_state_idle) {}

  void ompt_state_set(void *enter_frame, void *codeptr_ra);

  void ompt_state_clear();

  // target op callbacks
  void target_data_alloc_begin(int64_t device_id, void *TgtPtrBegin, size_t Size);

  void target_data_alloc_end(int64_t device_id, void *TgtPtrBegin, size_t Size);

  void target_data_submit_begin(int64_t device_id, void *HstPtrBegin, void *TgtPtrBegin, size_t Size);

  void target_data_submit_end(int64_t device_id, void *HstPtrBegin, void *TgtPtrBegin, size_t Size);

  void target_data_delete_begin(int64_t device_id, void *TgtPtrBegin); 

  void target_data_delete_end(int64_t device_id, void *TgtPtrBegin); 

  void target_data_retrieve_begin(int64_t device_id, void *HstPtrBegin, void *TgtPtrBegin, size_t Size); 

  void target_data_retrieve_end(int64_t device_id, void *HstPtrBegin, void *TgtPtrBegin, size_t Size); 

  void target_submit_begin(unsigned int num_teams=1);

  void target_submit_end(unsigned int num_teams=1);

  // target region callbacks
  void target_data_enter_begin(int64_t device_id);

  void target_data_enter_end(int64_t device_id);

  void target_data_exit_begin(int64_t device_id);

  void target_data_exit_end(int64_t device_id);

  void target_update_begin(int64_t device_id);

  void target_update_end(int64_t device_id);

  void target_begin(int64_t device_id);

  void target_end(int64_t device_id);

 private:
  void ompt_state_set_helper(void *enter_frame, void *codeptr_ra, int flags, int state);

  // begin/end target op marks
  void target_operation_begin();

  void target_operation_end();

  // begin/end target region marks
  uint64_t target_region_begin();

  uint64_t target_region_end();

 private:
  void *_enter_frame;
  void *_codeptr_ra;
  int _state;
}; 


extern thread_local OmptInterface ompt_interface; 

extern bool ompt_enabled;


#endif
