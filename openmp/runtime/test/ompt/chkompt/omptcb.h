// omptcb.c -- code for the interactions with the OpenMP library to verify
// 	the behavior of various callbacks

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <omp-tools.h>
#include "chkompt.h"


ompt_function_lookup_t my_lookup;
ompt_set_callback_t ompt_set_callback_fn;	// Address of routine to set callback
ompt_get_task_info_t ompt_get_task_info_fn;	// Address of routine to get task information

void (*validate_ptr)(const char *) = validate;
void (*ck_ra_)(const char *, int, const void *, int, char*) = ck_ra;

int in_implicit_task = 0;
hrtime_t	starttime;

// ------------------------------------------------------------------------
// inform the runtime that we will be using OMPT
// This routine is automatically invoked by the OpenMP runtime at
// its initialization.  It tells the library where to find:
// 	ompt_initialize -- which is invoked at the first entry to the runtime
//   and
// 	ompt_finalize -- which is invoked when the runtime shuts down
// ------------------------------------------------------------------------

void error_breakpoint() { }

ompt_start_tool_result_t *
ompt_start_tool
( 
	unsigned int omp_version,
	const char *runtime_version
)
{
	// fprintf(stderr, "ompt_start_tool invoked\n");
	static ompt_start_tool_result_t result = { 
	    ompt_initialize, 
	    ompt_finalize,  
	    ompt_data_none
	};
	return &result;
}

// ------------------------------------------------------------------------
// initialize upcall for OMPT
// ------------------------------------------------------------------------
int 
ompt_initialize
(
	ompt_function_lookup_t lookup,
	int initial_device_num,
	ompt_data_t *tool_data
)
{
	// fprintf(stderr, "ompt_initialize invoked\n");

	my_lookup = lookup;

	starttime = gethrtime();

	// look up the runtime entry points
	ompt_get_task_info_fn = (ompt_get_task_info_t) my_lookup("ompt_get_task_info");
#ifndef NO_CALLBACKS
	// look up two runtime entry points
	ompt_set_callback_fn = (ompt_set_callback_t) lookup("ompt_set_callback");

	// register callbacks to be notified about various events
	register_callbacks();
#endif
	return 1;
}

// ------------------------------------------------------------------------
// finalize upcall for OMPT -- nothing to do
// ------------------------------------------------------------------------
void 
ompt_finalize ( ompt_data_t *tool_data)
{
}


// ------------------------------------------------------------------------
// Register the various callbacks that will be tested
// ------------------------------------------------------------------------
char	*cb_names[] = {
	"illegal callback number",		//=0
	"ompt_callback_thread_begin",		//=1,
	"ompt_callback_thread_end",		//=2,
	"ompt_callback_parallel_begin",		//=3,
	"ompt_callback_parallel_end",		//=4,
	"ompt_callback_task_create",		//=5,
	"ompt_callback_task_schedule",		//=6,
	"ompt_callback_implicit_task",		//=7,
	"ompt_callback_target",			//=8,
	"ompt_callback_target_data_op",		//=9,
	"ompt_callback_target_submit",		//=10,
	"ompt_callback_control_tool",		//=11,
	"ompt_callback_device_initialize",	//=12,
	"ompt_callback_device_finalize",	//=13,
	"ompt_callback_device_load",		//=14,
	"ompt_callback_device_unload",		//=15,
	"ompt_callback_sync_region_wait",	//=16,
	"ompt_callback_mutex_released",		//=17,
	"ompt_callback_dependences",		//=18,
	"ompt_callback_task_dependence",	//=19,
	"ompt_callback_work",			//=20,
	"ompt_callback_master",			//=21,
	"ompt_callback_target_map",		//=22,
	"ompt_callback_sync_region",		//=23,
	"ompt_callback_lock_init",		//=24,
	"ompt_callback_lock_destroy",		//=25,
	"ompt_callback_mutex_acquire",		//=26,
	"ompt_callback_mutex_acquired",		//=27,
	"ompt_callback_nest_lock",		//=28,
	"ompt_callback_flush",			//=29,
	"ompt_callback_cancel",			//=30,
	"ompt_callback_reduction",		//=31,
	"ompt_callback_dispatch",		//=32
	NULL
	};

void
register_callbacks()
{
	int	ncallbacks = 0;

	ompt_set_result_t ret;

// Define a macro to set a callback
#define SetCallback(type,name) \
	ret = ompt_set_callback_fn ( type, (ompt_callback_t) name); \
	if ( (ret == ompt_set_error) || (ret == ompt_set_never) ) { \
	    fprintf(stderr, "    Note: %s (%2d) is never triggered in this implementation of OMPT (%d)\n", \
		cb_names[type], (int)type, (int)ret ); \
	} else if (ret == ompt_set_impossible) { \
	    fprintf(stderr, "    Note: %s (%2d) is impossible in this implementation of OMPT (%d)\n", \
		cb_names[type], (int)type, (int)ret ); \
	} else if ( (ret == ompt_set_sometimes) || (ret == ompt_set_sometimes_paired) ) { \
	    fprintf(stderr, "    Note: %s (%2d) may or may not be triggered in this implementation of OMPT (%d)\n", \
		cb_names[type], (int)type, (int)ret ); \
	} else { \
	    ncallbacks ++; \
	}

	// Callback for thread begin
	SetCallback(ompt_callback_thread_begin,ompt_thread_begin);

	// Callback for thread end
	SetCallback(ompt_callback_thread_end, ompt_thread_end);

	// Callback for parallel region begin
	SetCallback(ompt_callback_parallel_begin, ompt_parallel_begin);

	// Callback for parallel region end
	SetCallback(ompt_callback_parallel_end, ompt_parallel_end);

	// Callback for task creation
	SetCallback(ompt_callback_task_create, ompt_task_create);

	// Callback for task schedule
	SetCallback(ompt_callback_task_schedule, ompt_task_schedule);

	// Callback for implicit task creation
	SetCallback(ompt_callback_implicit_task, ompt_implicit_task);

	// Callback for target
	SetCallback(ompt_callback_target, ompt_targetcb);

	// Callback for target_data_op
	SetCallback(ompt_callback_target_data_op, ompt_target_data_op);

	// Callback for target submit
	SetCallback(ompt_callback_target_submit, ompt_target_submit);

	// Callback for control_tool
	SetCallback(ompt_callback_control_tool, ompt_control_tool);

	// Callback for device_initialize
	SetCallback(ompt_callback_device_initialize, ompt_device_initialize);

	// Callback for device_finalize
	SetCallback(ompt_callback_device_finalize, ompt_device_finalize);

	// Callback for device_load
	SetCallback(ompt_callback_device_load, ompt_device_load);

	// Callback for device_unload
	SetCallback(ompt_callback_device_unload, ompt_device_unload);

	// Callback for synchronization region wait
	SetCallback(ompt_callback_sync_region_wait, ompt_sync_region_wait);

	// Callback for mutex released
	SetCallback(ompt_callback_mutex_released, ompt_mutex_released);

	// Callback for dependences
	SetCallback(ompt_callback_dependences, ompt_dependences);

	// Callback for task_dependence
	SetCallback(ompt_callback_task_dependence, ompt_task_dependence);

	// Callback for work entry
	SetCallback(ompt_callback_work, ompt_work);

	// Callback for master region entry
	SetCallback(ompt_callback_master, ompt_master);

	// Callback for target map
	SetCallback(ompt_callback_target_map, ompt_target_map);

	// Callback for synchronization region
	SetCallback(ompt_callback_sync_region, ompt_sync_region);

	// Callback for lock init
	SetCallback(ompt_callback_lock_init, ompt_lock_init);

	// Callback for lock_destroy
	SetCallback(ompt_callback_lock_destroy, ompt_lock_destroy);

	// Callback for mutex acquire
	SetCallback(ompt_callback_mutex_acquire, ompt_mutex_acquire);

	// Callback for mutex acquired
	SetCallback(ompt_callback_mutex_acquired, ompt_mutex_acquired);

	// Callback for nest_lock
	SetCallback(ompt_callback_nest_lock, ompt_nest_lock);

	// Callback for flush
	SetCallback(ompt_callback_flush, ompt_flush);

	// Callback for cancel
	SetCallback(ompt_callback_cancel, ompt_cancel);

	// Callback for reduction
	SetCallback(ompt_callback_reduction, ompt_reduction);

	// Callback for dispatch
	SetCallback(ompt_callback_dispatch, ompt_dispatch);

	fprintf(stderr, "      %d other callbacks were set\n\n", ncallbacks);
}


// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// The various Callback routines
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// OMPT callback for implicit task creation 
// ------------------------------------------------------------------------
void
ompt_implicit_task
(
	ompt_scope_endpoint_t endpoint,
	ompt_data_t *parallel_data,
	ompt_data_t *task_data,
	unsigned int actual_parallelism,
	unsigned int index,
	int flags
)
{
	// trace the callback
	ck_ra("implicit_task_CB", 1, (const void*)1, (int)index,
	    (endpoint == ompt_scope_begin ? "[begin] " : "[end] ") );
	
	if (endpoint == ompt_scope_begin)  {
	    (*validate_ptr)("implicit task begin");
	    in_implicit_task = 1;
	} else if (endpoint == ompt_scope_end)  {
	    // (*validate_ptr)("implicit task end"); // can't validate
	    in_implicit_task = 0;
	} else {
	    abort();	// no others are defined
	}
}

// ------------------------------------------------------------------------
// OMPT callback for begin
// ------------------------------------------------------------------------
void
ompt_thread_begin
(
	ompt_thread_t thread_type,
	ompt_data_t *thread_data
)
{
	char *ctype = "unknown";
	switch (thread_type) {
	    case ompt_thread_initial:
		ctype = "[initial] ";
		break;
	    case ompt_thread_worker:
		ctype = "[worker] ";
		break;
	    case ompt_thread_other:
		ctype = "[other] ";
		break;
	    case ompt_thread_unknown:
		ctype = "[unknown] ";
		break;
	}
	ck_ra("thread_begin_CB", 1, (const void*)1, (int)thread_type, ctype);
}

// ------------------------------------------------------------------------
// OMPT callback for thread end
// ------------------------------------------------------------------------
void
ompt_thread_end
(
	ompt_data_t *thread_data
)
{
	ck_ra("thread_end_CB", 1, (const void*)1, 0, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for parallel region entry
// ------------------------------------------------------------------------
void
ompt_parallel_begin
(
	ompt_data_t *encountering_task_data,
	const ompt_frame_t *encountering_task_frame,
	ompt_data_t *parallel_data,
	unsigned int requested_parallelism,
	int flags,
	const void *codeptr_ra
)
{
	ck_ra("parallel_begin_CB", 0, codeptr_ra, 0, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for parallel region end
// ------------------------------------------------------------------------
void
ompt_parallel_end
(
	ompt_data_t *parallel_data,
	ompt_data_t *encountering_task_data,
	int flags,
	const void *codeptr_ra
)
{
	ck_ra("parallel_end_CB", 0, codeptr_ra, 0, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for task create
// ------------------------------------------------------------------------
void
ompt_task_create
(
	ompt_data_t *encountering_task_data,
	const ompt_frame_t *encountering_task_frame,
	ompt_data_t *new_task_data,
	int flags,
	int has_dependences,
	const void *codeptr_ra
)
{
	char	task_type[128];
	int is_initial;

	format_task_type(flags, task_type);

#if 0
	is_initial = flags & ompt_task_initial;
	if (!is_initial) {
	    ck_ra("task_create_CB", 0, codeptr_ra, 0, task_type);
	} else {
	    ck_ra("task_create_CB", 3, codeptr_ra, 0, task_type);
	}
#else
	ck_ra("task_create_CB", 0, codeptr_ra, 0, task_type);
#endif
}

// ------------------------------------------------------------------------
// OMPT callback for task schedule
// ------------------------------------------------------------------------
static char* ompt_task_status_t_values[] = {
	NULL,
	"[task_complete] ",       // 1
	"[task_yield] ",          // 2
	"[task_cancel] ",         // 3
	"[task_detach] ",         // 4
	"[task_early_fulfill] ",  // 5
	"[task_late_fulfill] ",   // 6
	"[task_switch] "          // 7
};
void
ompt_task_schedule
(
	ompt_data_t *prior_task_data,
	ompt_task_status_t prior_task_status,
	ompt_data_t *new_task_data
)
{
	ck_ra("task_schedule_CB", 1, (const void *)1, 0, ompt_task_status_t_values[prior_task_status] );
}

// ------------------------------------------------------------------------
// OMPT callback for ompt_target
// ------------------------------------------------------------------------
void
ompt_targetcb(
	ompt_target_t kind,
	ompt_scope_endpoint_t endpoint,
	int device_num,
	ompt_id_t task_data,
	ompt_id_t target_id,
	const void *codeptr_ra
)
{
	ck_ra("target_CB", 0, codeptr_ra, device_num, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for ompt_target_data_op
// ------------------------------------------------------------------------
void
ompt_target_data_op(
	ompt_id_t target_id,
	ompt_id_t host_op_id,
	ompt_target_data_op_t optype,
	void *src_addr,
	int src_device_num,
	void *dest_addr,
	int dest_device_num,
	size_t bytes,
	const void *codeptr_ra
)
{
	ck_ra("target data_op_CB", 0, codeptr_ra, src_device_num, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for ompt_target_submit
// ------------------------------------------------------------------------
void
ompt_target_submit(
	ompt_id_t target_id,
	ompt_id_t host_op_id,
	unsigned int requested_num_teams
)
{
	ck_ra("target_submit_CB", 1, (const void *)1, target_id, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for work
// ------------------------------------------------------------------------
void
ompt_work (
        ompt_work_t wstype,
        ompt_scope_endpoint_t endpoint,
        ompt_data_t *parallel_data,
        ompt_data_t *task_data,
        uint64_t count,
        const void *codeptr_ra
)
{
	char buffer[128];
	format_work_type(wstype, endpoint, buffer);

	ck_ra("work_CB", 0, codeptr_ra, (int)wstype, buffer);
}

// ------------------------------------------------------------------------
// OMPT callback for master
// ------------------------------------------------------------------------
void
ompt_master (
        ompt_scope_endpoint_t endpoint,
        ompt_data_t *parallel_data,
        ompt_data_t *task_data,
        const void *codeptr_ra
)
{
	ck_ra("master_CB", 0, codeptr_ra, (int)endpoint, 
	    (endpoint == ompt_scope_begin ? "[begin] " : "[end] ") );

}

// ------------------------------------------------------------------------
// OMPT callback for target_map
// ------------------------------------------------------------------------
void
ompt_target_map (
        ompt_id_t id,
        unsigned int nitems,
        void **host_adder,
        void **device_addr,
	size_t	*bytes,
	unsigned int *mapping_flags,
        const void *codeptr_ra
)
{
	ck_ra("target_map_CB", 0, codeptr_ra, (int) id, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for sync_region_wait
// ------------------------------------------------------------------------
void
ompt_sync_region_wait (
        ompt_sync_region_t kind,
        ompt_scope_endpoint_t endpoint,
        ompt_data_t *parallel_data,
        ompt_data_t *task_data,
        const void *codeptr_ra
)
{
	char buf[128];
	format_sync_type(kind, endpoint, buf);
	if (in_implicit_task) {
	    ck_ra("sync_region_wait_CB", 0, codeptr_ra, (int) kind, buf);
	} else {
	    ck_ra("sync_region_wait_CB", 2, codeptr_ra, (int) kind, buf);
	}
}

// ------------------------------------------------------------------------
// OMPT callback for sync_region
// ------------------------------------------------------------------------
void
ompt_sync_region (
        ompt_sync_region_t kind,
        ompt_scope_endpoint_t endpoint,
        ompt_data_t *parallel_data,
        ompt_data_t *task_data,
        const void *codeptr_ra
)
{
	char buf[128];
	format_sync_type(kind, endpoint, buf);
	if (in_implicit_task) {
	    ck_ra("sync_region_CB", 0, codeptr_ra, (int) kind, buf);
	} else {
	    ck_ra("sync_region_CB", 2, codeptr_ra, (int) kind, buf);
	}
}

// ------------------------------------------------------------------------
// OMPT callback for lock_init
// ------------------------------------------------------------------------
void
ompt_lock_init (
        ompt_mutex_t kind,
        unsigned int hint,
        unsigned int impl,
        ompt_wait_id_t wait_id,
        const void *codeptr_ra
)
{
	char buffer[128];
	format_lock_type (kind, buffer);
	ck_ra("lock_init_CB", 0, codeptr_ra, (int) kind, buffer);
}

// ------------------------------------------------------------------------
// OMPT callback for lock_destroy
// ------------------------------------------------------------------------
void
ompt_lock_destroy (
        ompt_mutex_t kind,
        ompt_wait_id_t wait_id,
        const void *codeptr_ra
)
{
	char buffer[128];
	format_lock_type (kind, buffer);
	ck_ra("lock_destroy_CB", 0, codeptr_ra, (int) kind, buffer);
}

// ------------------------------------------------------------------------
// OMPT callback for mutex_acquire
// ------------------------------------------------------------------------
void
ompt_mutex_acquire (
        ompt_mutex_t kind,
        unsigned int hint,
        unsigned int impl,
        ompt_wait_id_t wait_id,
        const void *codeptr_ra
)
{
	char buffer[128];
	format_lock_type (kind, buffer);
	ck_ra("mutex_acquire_CB", 0, codeptr_ra, (int) kind, buffer);
}

// ------------------------------------------------------------------------
// OMPT callback for mutex acquired
// ------------------------------------------------------------------------
void
ompt_mutex_acquired (
        ompt_mutex_t kind,
        ompt_wait_id_t wait_id,
        const void *codeptr_ra)
{
	char buffer[128];
	format_lock_type (kind, buffer);
	ck_ra("mutex_acquired_CB", 0, codeptr_ra, (int) kind, buffer);
}

// ------------------------------------------------------------------------
// OMPT callback for mutex released
// ------------------------------------------------------------------------
void
ompt_mutex_released(
        ompt_mutex_t kind,
        ompt_wait_id_t wait_id,
        const void *codeptr_ra)
{
	char buffer[128];
	format_lock_type (kind, buffer);
	ck_ra("mutex_released_CB", 0, codeptr_ra, (int) kind, buffer);
}

// ------------------------------------------------------------------------
// OMPT callback for dependences
// ------------------------------------------------------------------------
void
ompt_dependences (
        ompt_data_t *task_data,
	const ompt_dependence_t *deps,
	int ndeps
)
{
	ck_ra("dependences_CB", 1,(const void *)1, (int) ndeps, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for task_dependence
// ------------------------------------------------------------------------
void
ompt_task_dependence (
        ompt_data_t *src_task_data,
	ompt_data_t *sink_task_data
)
{
	ck_ra("task_dependence_CB", 1,(const void *)1, 0, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for nest_lock
// ------------------------------------------------------------------------
void
ompt_nest_lock (
        ompt_scope_endpoint_t endpoint,
        ompt_wait_id_t wait_id,
        const void *codeptr_ra
)
{
	ck_ra("nest_lock_CB", 0, codeptr_ra, (int) endpoint, 
	    (endpoint == ompt_scope_begin ? "[begin] " : "[end] ") );
}

// ------------------------------------------------------------------------
// OMPT callback for flush
// ------------------------------------------------------------------------
void
ompt_flush (
        ompt_data_t *thread_data,
        const void *codeptr_ra
)
{
	ck_ra("flush_CB", 0, codeptr_ra, 0, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for cancel
// ------------------------------------------------------------------------
static char* ompt_cancel_flag_t_values[] = {
	"[parallel] ",
	"[sections] ",
	"[loop] ",
	"[taskgroup] ",
	"[activated] ",
	"[detected] ",
	"[discarded_task] "
};
void
ompt_cancel (
        ompt_data_t *task_data,
        int flags,
        const void *codeptr_ra
)
{
	ck_ra("cancel_CB", 0, codeptr_ra, 0, ompt_cancel_flag_t_values[flags] );
}

// ------------------------------------------------------------------------
// OMPT callback for control_tool
// ------------------------------------------------------------------------
void
ompt_control_tool (
        uint64_t command,
        uint64_t modifier,
        void *arg,
        const void *codeptr_ra
)
{
	ck_ra("control_tool_CB", 0, codeptr_ra, 0, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for device_initialize
// ------------------------------------------------------------------------
void
ompt_device_initialize (
	int device_num,
	const char *type,
	ompt_device_t *device,
	ompt_function_lookup_t lookup,
	const char *documentation
)
{
	ck_ra("device_initialize_CB", 1, (const void*) 1, device_num, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for device_finalize
// ------------------------------------------------------------------------

void
ompt_device_finalize(
	int device_num
)
{
	ck_ra("device_finalize_CB", 1, (const void*) 1, device_num, NULL);
}


// ------------------------------------------------------------------------
// OMPT callback for device_load
// ------------------------------------------------------------------------

void
ompt_device_load(
	int device_num,
	const char *filename,
	int64_t offset_in_file,
	void *vma_in_file,
	size_t bytes,
	void *host_addr,
	void *device_addr,
	uint64_t module_id
)
{
	ck_ra("device_load_CB", 1, (const void*) 1, device_num, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for device_unload
// ------------------------------------------------------------------------
void
ompt_device_unload (
	int device_num,
	uint64_t module_id
)
{
	ck_ra("device_unload_CB", 1, (const void*) 1, device_num, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for dispatch
// ------------------------------------------------------------------------
void
ompt_reduction (
	ompt_sync_region_t kind,
	ompt_scope_endpoint_t endpoint,
	ompt_id_t parallel_id,
	ompt_id_t task_id,
        const void *codeptr_ra
)
{
	ck_ra("reduction_CB", 0, codeptr_ra, (int)kind, NULL);
}

// ------------------------------------------------------------------------
// OMPT callback for dispatch
// ------------------------------------------------------------------------
void
ompt_dispatch (
        ompt_data_t *parallel_data,
        ompt_data_t *task_data,
        ompt_dispatch_t kind,
        ompt_data_t instance
)
{
	ck_ra("dispatch_CB", 1, (const void*)1, (int)kind, NULL);
}

// ------------------------------------------------------------------------
// format_sync_type -- convert a synchonization type and endpoint to a string
// ------------------------------------------------------------------------
void
format_sync_type(ompt_sync_region_t type, ompt_scope_endpoint_t endpoint, char *buffer)
{
	char *ctype = "unknown";
	char *progress = buffer;

	if (endpoint == ompt_scope_begin) {
	    progress += sprintf(progress, "[begin ");
	} else {
	    progress += sprintf(progress, "[end ");
	}
	switch (type)
	    {
		case ompt_sync_region_barrier:
		    ctype = "barrier";
		    break;
		case ompt_sync_region_barrier_implicit:
		    ctype = "barrier-implicit";
		    break;
		case ompt_sync_region_barrier_explicit:
		    ctype = "barrier-explicit";
		    break;
		case ompt_sync_region_barrier_implementation:
		    ctype = "barrier-implementation";
		    break;
		case ompt_sync_region_taskwait:
		    ctype = "taskwait";
		    break;
		case ompt_sync_region_taskgroup:
		    ctype = "taskgroup";
		    break;
		case ompt_sync_region_reduction:
		    ctype = "reduction";
		    break;
	    }
	progress += sprintf(progress, "%s] ", ctype);
}

// ------------------------------------------------------------------------
// format_work_type -- convert a work type and endpoint to a string
// ------------------------------------------------------------------------
void
format_work_type(ompt_work_t type, ompt_scope_endpoint_t endpoint, char *buffer)
{
	char *ctype = "unknown";
	char *progress = buffer;
	if (endpoint == ompt_scope_begin) {
	    progress += sprintf(progress, "[begin ");
	} else {
	    progress += sprintf(progress, "[end ");
	}
	switch (type)
	    {
		case ompt_work_loop:
		    ctype = "loop";
		    break;
		case ompt_work_sections:
		    ctype = "sections";
		    break;
		case ompt_work_single_executor:
		    ctype = "single_executor";
		    break;
		case ompt_work_single_other:
		    ctype = "single_other";
		    break;
		case ompt_work_workshare:
		    ctype = "worksharc";
		    break;
		case ompt_work_distribute:
		    ctype = "distribute";
		    break;
		case ompt_work_taskloop:
		    ctype = "taskloop";
		    break;
	   }
	progress += sprintf(progress, "%s] ", ctype);
}

// ------------------------------------------------------------------------
// format_task_type -- convert a task type to a string
// ------------------------------------------------------------------------
void
format_task_type(int type, char *buffer)
{
	char *progress = &buffer[1];
	buffer[0] = '[';
	if (type & ompt_task_initial)
		  progress += sprintf(progress, "initial");
	if (type & ompt_task_implicit)
		  progress += sprintf(progress, "implicit");
	if (type & ompt_task_explicit)
		  progress += sprintf(progress, "explicit");
	if (type & ompt_task_target)
		  progress += sprintf(progress, "target");
	if (type & ompt_task_undeferred)
		  progress += sprintf(progress, "|undeferred");
	if (type & ompt_task_untied)
		  progress += sprintf(progress, "|untied");
	if (type & ompt_task_final)
		  progress += sprintf(progress, "|final");
	if (type & ompt_task_mergeable)
		  progress += sprintf(progress, "|mergeable");
	if (type & ompt_task_merged)
		  progress += sprintf(progress, "|merged");
	progress += sprintf(progress, "] ");
}

// ------------------------------------------------------------------------
// format_lock_type -- convert a lock type to a string
// ------------------------------------------------------------------------
void
format_lock_type(ompt_mutex_t type, char *buffer)
{
	char *ctype = "unknown";

	switch (type) {
	    case ompt_mutex_lock:
		ctype = "mutex";
		break;
	    case ompt_mutex_test_lock:
		ctype = "mutex_test";
		break;
	    case ompt_mutex_nest_lock:
		ctype = "mutex_nest";
		break;
	    case ompt_mutex_test_nest_lock:
		ctype = "mutex_test_nest";
		break;
	    case ompt_mutex_critical:
		ctype = "mutex_critical";
		break;
	    case ompt_mutex_atomic:
		ctype = "mutex_atomic";
		break;
	    case ompt_mutex_ordered:
		ctype = "mutex_ordered";
		break;
	}
	sprintf(buffer, "[%s] ", ctype);

}


// ------------------------------------------------------------------------
// ck_ra -- invoked from various callbacks
//	check that the return address pointer is non-NULL
//	ckra parameter is:
//		0 to report an error if NULL
//		1 if address is not supplied in the callback
//		2 if in the implicit task
//		3 if xxxx
//
// ------------------------------------------------------------------------
void 
ck_ra(const char * type, int ckra, const void *ra, int param, char  *desc)
{
	int	threadnum;
	char	buf[512];
	threadnum = omp_get_thread_num();
	if ( (ckra == 0) && (ra == NULL) ) {
	    sprintf( buf,
		"%25s -- ERROR  -- %sthread %3d, param = %d, codeptr_ra == NULL\n",
		type, (desc != NULL? desc : ""), threadnum, param );
	    ts_write (buf);
	    error_breakpoint();

#pragma omp atomic update
	    nfails ++;
	} else {
#ifdef TRACE_ALL
	    if (ckra == 0) {
		sprintf( buf,
		    "%25s OK ck_ra  -- %sthread %3d, param = %d codeptr_ra = %p\n", 
		    type, (desc != NULL? desc : ""), threadnum, param, ra );
	    } else if (ckra == 1) {
		sprintf( buf,
		    "%25s OK ck_ra  -- %sthread %3d, param = %d\n", 
		    type, (desc != NULL? desc : ""), threadnum, param );
	    } else if (ckra == 2 ) {
		sprintf( buf,
		    "%25s OK ck_ra  -- %sthread %3d, param = %d codeptr_ra = %p -- in implicit task\n", 
		    type, (desc != NULL? desc : ""), threadnum, param, ra );
#if 0
	    } else if (ckra == 3 ) {
		sprintf( buf,
		    "%25s OK ck_ra  -- %sthread %3d, param = %d codeptr_ra = %p -- initial task create\n", 
		    type, (desc != NULL? desc : ""), threadnum, param, ra );
#endif
	    } else {
		fprintf(stderr, "Ooops, INTERNAL ERROR -- invalid ckra parameter = %d\n", ckra);
	    }
	    ts_write (buf);
#endif
	}

}


// ------------------------------------------------------------------------
// validate
// 	delay a bit
// 	ask for the caller's frame
//	    check that its exit_frame pointer is non-NULL, and flag is non-zero
//	    check that its enter_frame pointer is NULL, and flag is zero
// 	ask for the caller's ancestors' frame
//	    check that its exit_frame pointer is non-NULL, and flag is non-zero
// 	delay a varying amount, depending on thread number to desynchonize the threads
// ------------------------------------------------------------------------
void 
validate(const char *type) 
{
	int thread_num;
	ompt_frame_t *task_frame; 
	ompt_frame_t *parent_task_frame; 
	char	buf[256];

#ifdef	RUN_SKEW
	(*skew_delay_ptr)(1);
#endif

	ompt_get_task_info_fn
		(
		 0, // ancestor_level
		 NULL, // flags
		 NULL, // task_data
		 &task_frame,
		 NULL, // parallel_data
		 &thread_num
		);

	// Check for failure
	if (task_frame == NULL) {
	    sprintf( buf,
		"%25s -- ERROR  -- thread %3d task_frame = NULL\n",
		type, thread_num);
	    ts_write (buf);
	    error_breakpoint();

#pragma omp atomic update
	    nfails ++;

	} else if (task_frame->exit_frame.ptr == NULL) {
	    error_breakpoint();
	    sprintf( buf,
		"%25s -- ERROR  -- thread %3d exit_frame.ptr = NULL\n",
		type, thread_num);
	    ts_write (buf);

#pragma omp atomic update
	    nfails ++;

	} else if (task_frame->exit_frame_flags == 0) {
	    sprintf( buf,
		"%25s -- ERROR  -- thread %3d exit_frame.flags = 0\n",
		type, thread_num);
	    ts_write (buf);
	    error_breakpoint();

#pragma omp atomic update
	    nfails ++;

	} else if (task_frame->enter_frame.ptr != NULL) {
	    sprintf( buf,
		"%25s -- ERROR  -- thread %3d enter_frame.ptr != NULL\n",
		type, thread_num);
	    ts_write (buf);
	    error_breakpoint();

#pragma omp atomic update
	    nfails ++;

	} else if (task_frame->enter_frame_flags != 0) {
	    sprintf( buf,
		"%25s -- ERROR  -- thread %3d enter_frame.flags = 0x%02x != 0\n",
		type, thread_num, task_frame->enter_frame_flags);
	    ts_write (buf);
	    error_breakpoint();

#pragma omp atomic update
	    nfails ++;

	// Now check the enter_frame for the ancestor
	        ompt_get_task_info_fn
	                (
	                 1, // ancestor_level
	                 NULL, // flags
	                 NULL, // task_data
	                 &parent_task_frame,
	                 NULL, // parallel_data
	                 &thread_num
	                );

	        if (parent_task_frame == NULL) {
	            sprintf( buf,
	                "%25s -- ERROR  -- thread %3d parent_task_frame = NULL\n",
	                type, thread_num);
	    	    ts_write (buf);
		    error_breakpoint();

#pragma omp atomic update
		    nfails ++;
	        } else if (parent_task_frame->enter_frame.ptr == NULL) {
	            sprintf( buf,
	                "%25s -- ERROR  -- thread %3d parent enter_frame.ptr = NULL\n",
	                type, thread_num);
	    	    ts_write (buf);
		    error_breakpoint();

#pragma omp atomic update
	            nfails ++;
	        } else if (parent_task_frame->enter_frame_flags == 0) {
	            sprintf( buf,
	                "%25s -- ERROR  -- thread %3d parent enter_frame_flags = 0\n",
	                type, thread_num);
	    	    ts_write (buf);
		    error_breakpoint();

#pragma omp atomic update
	            nfails ++;
	        }


	} else {
#ifdef TRACE_ALL
	    sprintf( buf,
		"%25s OK return -- thread %3d exit_frame.ptr = %p  flags = 0x%02x\n", 
		type, thread_num, task_frame->exit_frame.ptr, task_frame->exit_frame_flags);
	    ts_write (buf);
#endif
	}
#ifdef	RUN_SKEW
	(*skew_delay_ptr)(thread_num);
#endif
}


// ------------------------------------------------------------------------
// ts_write -- write error (or log) to stderr, with a timestamp
// 	if NOTIMESTAMP is defined, don't write the timestamp
// ------------------------------------------------------------------------
void
ts_write (char *message)
{
	hrtime_t delta;
	char	buf[512];
	int	sec;
	int	nsec;
#ifdef NO_TIMESTAMPS
	fwrite (message, strlen(message), 1, stderr );
#else
	delta = gethrtime() - starttime;
	sec = delta / 1000000000;
	nsec = delta % 1000000000;

	sprintf(buf, "%4d.%09d: %s",
	    sec, nsec, message);
	fwrite (buf, strlen(buf), 1, stderr );
#endif
}

hrtime_t
gethrtime()
{
	return ( (hrtime_t) (omp_get_wtime() * 1.0E09) );
}
