// chkompt.h -- definitions for chkompt, test to verify the OMPT interface
//	Contributed by the HPCToolkit team, Rice University, 2019
//

#define N 40   // iteration count
#define NUMTHREADS 4

// The next three lines define how the output is written.
// If TRACE_ALL is defined, all events are traced, whether an ERROR is detected or not.
// If NO_TIMESTAMPS is defined, the trace will not have timestamps; otherwise it will.
// If NO_CALLBACKS is defined, no callbacks will be set, so none will appear in the trace.
// If NO_NONEST is defined, do not run the testtriple_nonest subtest

#define TRACE_ALL
// #define NO_TIMESTAMPS
// #define NO_CALLBACKS
// #define NO_NONEST

#define RUN_SKEW
#define NSKEW 1000000

// ------------------------------------------------------------------------
// Subtests for various parts of the OMPT interface
// ------------------------------------------------------------------------
void	testparallel();
void	testparallelfor();
void	teststatic();
void	testdynamic();
void	testguided();
void	testsections();
void	testparallelsections();
void	testtasks();
void	testtriple_nest();
void	testtriple_nonest();
void	testtriple();
void	reductiontest();
void	lockcbtest();

// Utility routines to validate the data and delay threads
void	skew_delay(int);
void	format_sync_type(ompt_sync_region_t, ompt_scope_endpoint_t endpoint, char *);
void	format_work_type(ompt_work_t, ompt_scope_endpoint_t endpoint, char *);
void	format_task_type(int, char *);
void	format_lock_type(ompt_mutex_t, char *);
void	ck_ra(const char *, int, const void*, int, char*);
void	validate(const char *);
void	delay(int);

// pointers to those routines
void	(*skew_delay_ptr)(int);
void	(*ck_ra_ptr)(const char *, int, const void *, char *);
void	(*validate_ptr)(const char *);
void	(*delay_ptr)(int);

// OpenMP library invocation routines
ompt_function_lookup_t my_lookup;
ompt_start_tool_result_t * ompt_start_tool(unsigned int, const char *);
void	ompt_finalize();
int	ompt_initialize(ompt_function_lookup_t, int, ompt_data_t *);
void	register_callbacks();

// OpenMP library functions, looked up at initialization
ompt_set_callback_t ompt_set_callback_fn;
ompt_get_task_info_t ompt_get_task_info_fn;

// OpenMP callback routines
void	ompt_implicit_task(ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*, unsigned int, unsigned int, int);
void	ompt_thread_begin(ompt_thread_t, ompt_data_t *);
void	ompt_thread_end(ompt_data_t *);
void	ompt_parallel_begin(ompt_data_t *, const ompt_frame_t *, ompt_data_t *, unsigned int, int, const void *);
void	ompt_parallel_end(ompt_data_t *, ompt_data_t *, int, const void *);
void	ompt_task_create(ompt_data_t *, const ompt_frame_t *, ompt_data_t *, int, int, const void *);
void	ompt_task_schedule(ompt_data_t *, ompt_task_status_t, ompt_data_t *);
void	ompt_implicit_task(ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*, unsigned int, unsigned int, int);

void	ompt_targetcb (ompt_target_t, ompt_scope_endpoint_t, int, ompt_id_t, ompt_id_t, const void *);

void	ompt_target_data_op( ompt_id_t, ompt_id_t, ompt_target_data_op_t, void *, int, void *, int, size_t, const void *);
void	ompt_target_submit (ompt_id_t, ompt_id_t, unsigned int);

void	ompt_control_tool(uint64_t, uint64_t, void *, const void *);
void	ompt_device_initialize(int, const char*, ompt_device_t *, ompt_function_lookup_t, const char *);
void	ompt_device_finalize(int);
void	ompt_device_load(int, const char*, int64_t, void*, size_t, void*, void*,uint64_t);
void	ompt_device_unload(int, uint64_t);

void	ompt_sync_region_wait(ompt_sync_region_t, ompt_scope_endpoint_t, ompt_data_t *, ompt_data_t *, const void *);
void	ompt_mutex_released(ompt_mutex_t, ompt_wait_id_t, const void *);

void	ompt_dependences(ompt_data_t*, const ompt_dependence_t *, int);
void	ompt_task_dependence(ompt_data_t*, ompt_data_t *);

void	ompt_work(ompt_work_t, ompt_scope_endpoint_t, ompt_data_t *, ompt_data_t *, uint64_t, const void *);
void	ompt_master( ompt_scope_endpoint_t, ompt_data_t *, ompt_data_t *, const void *);
void	ompt_target_map(ompt_id_t, unsigned int, void **, void **, size_t*, unsigned int *, const void *);
void	ompt_sync_region(ompt_sync_region_t, ompt_scope_endpoint_t, ompt_data_t *, ompt_data_t *, const void *);
void	ompt_lock_init( ompt_mutex_t, unsigned int, unsigned int, ompt_wait_id_t, const void *);
void	ompt_lock_destroy(ompt_mutex_t, ompt_wait_id_t, const void *);
void	ompt_mutex_acquire( ompt_mutex_t, unsigned int, unsigned int, ompt_wait_id_t, const void *);
void	ompt_mutex_acquired(ompt_mutex_t, ompt_wait_id_t, const void *);
void	ompt_nest_lock(ompt_scope_endpoint_t, ompt_wait_id_t, const void *);
void	ompt_flush(ompt_data_t *, const void *);
void	ompt_cancel(ompt_data_t *, int, const void *);
void	ompt_reduction();
void	ompt_dispatch(ompt_data_t *, ompt_data_t *, ompt_dispatch_t, ompt_data_t instance);

// Timing routines
typedef long long      hrtime_t;        // time in nanoseconds
hrtime_t	gethrtime();
hrtime_t	gethrvtime();
hrtime_t	gethrustime();

void	ts_write(char *);

// Data -- keeping track of failures
int     nfails;

hrtime_t	starttime;

