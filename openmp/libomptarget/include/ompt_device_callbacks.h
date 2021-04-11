#ifndef ompt_device_callback_h
#define ompt_device_callback_h

//****************************************************************************
// local includes
//****************************************************************************

#include <ompt.h>



//****************************************************************************
// macros
//****************************************************************************

#define unwrap_fptr(x) ((void *) (uint64_t) x)



//****************************************************************************
// types
//****************************************************************************

class ompt_device_callbacks_t { 
 public:

  virtual void device_initialize
  (
   int device_num,
   const char *type
   ) {
    if (ompt_callback_device_initialize_fn) {
      ompt_callback_device_initialize_fn
	(device_num, type, lookup_device(device_num),
	 lookup, documentation);
    }
  };

  virtual void ompt_callback_device_finalize
    (
     int device_num
     ) {
    if (ompt_callback_device_finalize_fn) {
      ompt_callback_device_finalize_fn(device_num);
    }
  };


  virtual void ompt_callback_device_load
    (
     int device_num,
     const char *filename,
     int64_t offset_in_file,
     void *vma_in_file,
     size_t bytes,
     void *host_addr,
     void *device_addr,
     uint64_t module_id
     ) {
    if (ompt_callback_device_load_fn) {
      ompt_callback_device_load_fn
	(device_num, filename, offset_in_file, vma_in_file,
	 bytes, host_addr, device_addr, module_id);
    }
  };

  
  virtual void ompt_callback_device_unload
  (
    int device_num,
    uint64_t module_id
   ) {
    if (ompt_callback_device_unload_fn) {
      ompt_callback_device_unload_fn
	(device_num, module_id);
    }
  };


  virtual void ompt_callback_target_data_op_emi
    (
     ompt_scope_endpoint_t endpoint,
     ompt_data_t *target_task_data,
     ompt_data_t *target_data,
     ompt_id_t *host_op_id,
     ompt_target_data_op_t optype,
     void *src_addr,
     int src_device_num,
     void *dest_addr,
     int dest_device_num,
     size_t bytes,
     const void *codeptr_ra
     ) {
    if (ompt_callback_target_data_op_emi_fn) {
      ompt_callback_target_data_op_emi_fn
	(endpoint, target_task_data, target_data, host_op_id, optype,
	 src_addr, src_device_num, dest_addr, dest_device_num, bytes,
	 codeptr_ra);
    }
  };


  virtual void ompt_callback_target_data_op
    (
     ompt_scope_endpoint_t endpoint,
     ompt_id_t target_id,
     ompt_id_t host_op_id,
     ompt_target_data_op_t optype,
     void *src_addr,
     int src_device_num,
     void *dest_addr,
     int dest_device_num,
     size_t bytes,
     const void *codeptr_ra
     ) {
    if (ompt_callback_target_data_op_fn) {
      ompt_callback_target_data_op_fn
	(endpoint, target_id, host_op_id, optype, src_addr, src_device_num,
	 dest_addr, dest_device_num, bytes, codeptr_ra);
    }
  };


  virtual void ompt_callback_target_emi
    (
     ompt_target_t kind,
     ompt_scope_endpoint_t endpoint,
     int device_num,
     ompt_data_t *task_data,
     ompt_data_t *target_task_data,
     ompt_data_t *target_data,
     const void *codeptr_ra
     ) {
    if (ompt_callback_target_emi_fn) {
      ompt_callback_target_emi_fn
	(kind, endpoint, device_num, task_data, target_task_data,
	 target_data, codeptr_ra);
    }
  };


  virtual void ompt_callback_target
    (
     ompt_target_t kind,
     ompt_scope_endpoint_t endpoint,
     int device_num,
     ompt_data_t *task_data,
     ompt_id_t target_id,
     const void *codeptr_ra
     ) {
    if (ompt_callback_target_fn) {
      ompt_callback_target_fn
	(kind, endpoint, device_num, task_data,
	 target_id, codeptr_ra);
    }
  };


  virtual void ompt_callback_target_map_emi
    (
     ompt_data_t *target_data,
     unsigned int nitems,
     void **host_addr,
     void **device_addr,
     size_t *bytes,
     unsigned int *mapping_flags,
     const void *codeptr_ra
     ) {
    if (ompt_callback_target_map_emi_fn) {
      ompt_callback_target_map_emi_fn
	(target_data, nitems, host_addr, device_addr,
	 bytes, mapping_flags, codeptr_ra);
    }
  };


  virtual void ompt_callback_target_map
    (
     ompt_id_t target_id,
     unsigned int nitems,
     void **host_addr,
     void **device_addr,
     size_t *bytes,
     unsigned int *mapping_flags,
     const void *codeptr_ra
     ) {
    if (ompt_callback_target_map_fn) {
      ompt_callback_target_map_fn
	(target_id, nitems, host_addr, device_addr,
	 bytes, mapping_flags, codeptr_ra);
    }
  };


  virtual void ompt_callback_target_submit_emi
    (
     ompt_scope_endpoint_t endpoint,
     ompt_data_t *target_data,
     ompt_id_t *host_op_id,
     unsigned int requested_num_teams
     ) {
    if (ompt_callback_target_submit_emi_fn) {
      ompt_callback_target_submit_emi_fn
	(endpoint, target_data, host_op_id, requested_num_teams);
    }
  };


  virtual void ompt_callback_target_submit
    (
     ompt_scope_endpoint_t endpoint,
     ompt_id_t target_id,
     ompt_id_t host_op_id,
     unsigned int requested_num_teams
     ) {
    if (ompt_callback_target_submit_fn) {
      ompt_callback_target_submit_fn
	(endpoint, target_id, host_op_id, requested_num_teams); 
    }
  };


#if 0
  ompt_device_callbacks_t () {
#define init_name(name) name ## _fn = 0; 
  FOREACH_OMPT_TARGET_CALLBACK(init_name)
#undef init_name
  };
#endif

  
  void register_callbacks(ompt_function_lookup_t lookup) {
#define ompt_bind_callback(fn)					\
    fn ## _fn = (fn ## _t ) lookup(#fn);			\
    DP("OMPT: class bound %s=%p\n", #fn, unwrap_fptr(fn ## _fn));
    FOREACH_OMPT_TARGET_CALLBACK(ompt_bind_callback)
#undef ompt_bind_callback
  };

  
 private:


#define declare_name(name) name ## _t name ## _fn;
  FOREACH_OMPT_TARGET_CALLBACK(declare_name)
#undef declare_name

  static ompt_interface_fn_t lookup(const char *interface_function_name);
  static ompt_device_t *lookup_device(int device_num);
  static const char *documentation;
};


extern ompt_device_callbacks_t ompt_interface;

#endif
