// RUN: %libomp-compile && env OMP_CANCELLATION=true %libomp-run | %sort-threads | FileCheck %s
#define __STDC_FORMAT_MACROS

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <omp-tools.h>

#include "chkompt.h"

int	nfails = 0;

void (*skew_delay_ptr)(int) = skew_delay;
void (*delay_ptr)(int) = delay;

// ------------------------------------------------------------------------
// Main program -- drive various tests
// ------------------------------------------------------------------------

int main(int argc, char **argv)
{
	char	buf[256];

	starttime = gethrtime();
	// fprintf(stderr, "main invoked\n");

	// Set thread count; causes the initialization of the OMPT code
	omp_set_num_threads(NUMTHREADS);

	// test lock callbacks
	lockcbtest();
	(*delay_ptr)(10);

	// test reduction
	reductiontest();
	(*delay_ptr)(10);

	// Test frames for multiple loops in a single parallel region
	testparallel();
	(*delay_ptr)(10);

	// Test frames for independent parallel for loops with static scheduling
	testparallelfor();
	(*delay_ptr)(10);

	// test parallel sections
	testparallelsections();
	(*delay_ptr)(10);

	// test explicit tasks
	testtasks();
	(*delay_ptr)(10);

	// test triply nested loops
	testtriple_nest();
	(*delay_ptr)(10);
#ifndef NO_NONEST
	testtriple_nonest();
	(*delay_ptr)(10);
#endif

	// Check for failures
	if (nfails != 0 ) {
	    sprintf(buf, "\n       FAILURE:\n\t%d ERROR%s detected\n\n",
		nfails,
		nfails == 1 ? "" : "s" );
	    ts_write(buf);
	    printf("\n       FAILURE:\n\t%d ERROR%s detected\n\n",
		nfails,
		nfails == 1 ? "" : "s" );
		
	    exit(1);
	} else {
	    ts_write("\n       No failures\n\n");
	    printf("No failures\n");
	    exit(0);
	}
// CHECK: No failures
}

// ------------------------------------------------------------------------
// Test "omp parallel" with "omp for" loops with various schedules
// ------------------------------------------------------------------------
void testparallel()
{
	int i;
	ts_write("\n                  starting testparallel\n\n");
#pragma omp parallel private(i)
	{
	    (*validate_ptr)("parallel start");
#pragma omp master
	ts_write("\n                  starting for\n\n");
#pragma omp for
	    for(i = 0; i < N; i++) (*validate_ptr)("for");
#pragma omp master
	    (*delay_ptr)(10);

#pragma omp master
	ts_write("\n                  starting for static\n\n");
#pragma omp for schedule(static)
	    for(i = 0; i < N; i++) (*validate_ptr)("for schedule(static)");
#pragma omp master
	    (*delay_ptr)(10);

#pragma omp master
	ts_write("\n                  starting for dynamic\n\n");
#pragma omp for schedule(dynamic)
	    for(i = 0; i < N; i++) (*validate_ptr)("for schedule(dynamic)");
#pragma omp master
	    (*delay_ptr)(10);

#pragma omp master
	ts_write("\n                  starting for guided\n\n");
#pragma omp for schedule(guided)
	    for(i = 0; i < N; i++) (*validate_ptr)("for schedule(guided)");
#pragma omp master
	    (*delay_ptr)(10);

	    (*validate_ptr)("parallel end");
	}
}

// ------------------------------------------------------------------------
// Test "omp parallel for" loops with various schedules
// ------------------------------------------------------------------------
void
testparallelfor()
{
	teststatic();
	(*delay_ptr)(10);

	testdynamic();
	(*delay_ptr)(10);

	testguided();
	(*delay_ptr)(10);

	testsections();
	(*delay_ptr)(10);
}

// ------------------------------------------------------------------------
// Test "omp parallel for" loops with various schedules
// ------------------------------------------------------------------------
void teststatic()
{
	int i;
	ts_write("\n                  starting teststatic\n\n");
#pragma omp parallel for schedule(static) private(i)
	for(i = 0; i < N; i++) (*validate_ptr)("parallel for static");
}


void testdynamic()
{
	int i;
	ts_write("\n                  starting testdynamic\n\n");
#pragma omp parallel for schedule(dynamic) private(i)
	for(i = 0; i < N; i++) (*validate_ptr)("parallel for dynamic");
}


void testguided()
{
	int i;
	ts_write("\n                  starting testguided\n\n");
#pragma omp parallel for schedule(guided) private(i)
	for(i = 0; i < N; i++) (*validate_ptr)("parallel for guided");
}

// ------------------------------------------------------------------------
// Test "omp sections"
// ------------------------------------------------------------------------
void testsections()
{
	    ts_write("\n                  starting testsections\n\n");
#pragma omp parallel
	{
#pragma omp sections
	    {
#pragma omp section
		{
		    (*validate_ptr)("omp section 1");
#ifdef	RUN_SKEW
		    (*skew_delay_ptr)(1);
#endif
		}
#pragma omp section
		{
		    (*validate_ptr)("omp section 2");
#ifdef	RUN_SKEW
		    (*skew_delay_ptr)(2);
#endif
		}
#pragma omp section
		{
		    (*validate_ptr)("omp section 3");
#ifdef	RUN_SKEW
		    (*skew_delay_ptr)(3);
#endif
		}
	    }
	}
}


void testparallelsections()
{
	    ts_write("\n                  starting testparallelsections\n\n");
#pragma omp parallel sections num_threads(NUMTHREADS)
	{
#pragma omp section
	      (*validate_ptr)("omp parallel section 1");
#pragma omp section
	      (*validate_ptr)("omp parallel section 2");
#pragma omp section
	      (*validate_ptr)("omp parallel section 3");
	}
}

void testtasks()
{
	    ts_write("\n                  starting testtasks\n\n");
#pragma omp parallel
	{
#pragma omp single
	    {
#pragma omp task
	      (*validate_ptr)("omp task 1");
#pragma omp task
	      (*validate_ptr)("omp task 2");
#pragma omp task
	      (*validate_ptr)("omp task 3");
#pragma omp task
	      (*validate_ptr)("omp task 4");
#pragma omp task
	      (*validate_ptr)("omp task 5");
#pragma omp task
	      (*validate_ptr)("omp task 6");
#pragma omp task
	      (*validate_ptr)("omp task 7");
#pragma omp task
	      (*validate_ptr)("omp task 8");
#pragma omp task
	      (*validate_ptr)("omp task 9");
	    }
	}
}


void	loop0();
void	loop1();
void	loop2();
void	loop3();

// testtriple_nest -- test a triply-nested set of loops, with nesting enabled
//
void testtriple_nest()
{
	ts_write("\n                  starting testtriple_nest\n\n");

	// Set omp_max_active_levels, to allow nested loops
	omp_set_max_active_levels(5);

	// now invoke the triply-nested loop
	testtriple();
}

// testtriple_nonest -- test a triply-nested set of loops, with nesting disabled
//
void testtriple_nonest()
{
	ts_write("\n                  starting testtriple_nonest\n\n");

	// Set omp_max_active_levels, to 1, disallowing nesting
	omp_set_max_active_levels(1);

	// now invoke the triply-nested loop
	testtriple();
}

// testtriple --the actual code, triply nested in source
void
testtriple()
{
#pragma omp parallel num_threads(2)
  {
    loop0();
#pragma omp parallel num_threads(2)
    {
      loop1();
#pragma omp parallel num_threads(3)
      {
	loop2();
#pragma omp parallel num_threads(3)
	{
	  loop3();
	}
      }
    }
  }
// 	omp_set_num_threads(NUMTHREADS);
}

#define ITERATIONS 100000000

void
form_label(char *buffer, char *label)
{
	int level, thread, pthread;

	level = omp_get_level();
	thread = omp_get_thread_num();
	pthread = omp_get_ancestor_thread_num(level);
	sprintf(buffer, "Begin %s t=%d l=%d pt=%d", label, thread, level, pthread);
}

void
loop0()
{
	int j;
	char buf[100];
	form_label( buf, "loop0");

	(*validate_ptr)(buf);
	for(j=0;j<ITERATIONS;j+=2) j--;

	strncpy (buf, "End   ", 6);
	(*validate_ptr)(buf);
}

void
loop1()
{
	int j;
	char buf[100];
	form_label( buf, "loop1");

	(*validate_ptr)(buf);
	for(j=0;j<ITERATIONS;j+=2) j--;

	strncpy (buf, "End   ", 6);
	(*validate_ptr)(buf);
}

void
loop2()
{
	int j;
	char buf[100];
	form_label( buf, "loop2");

	(*validate_ptr)(buf);
	for(j=0;j<ITERATIONS;j+=2) j--;

	strncpy (buf, "End   ", 6);
	(*validate_ptr)(buf);
}

void
loop3()
{
	int j;
	char buf[100];
	form_label( buf, "loop3");

	(*validate_ptr)(buf);
	for(j=0;j<ITERATIONS;j+=2) j--;

	strncpy (buf, "End   ", 6);
	(*validate_ptr)(buf);
}

// reductiontest -- check for appropriate callbacks
//
void
reductiontest()
{
	int sum, i;

	ts_write("\n                  starting reductiontest\n\n");
	sum = 0;

#pragma omp parallel for reduction(+:sum)
	for(i = 0; i < N; i++) {
	    sum += i;
	    (*validate_ptr)("reductiontest");
	}
}

// -----------------------------------------------------------
// lockcbtest -- make various omp lock calls and verify that
// the code pointers are plausible
//
void
lockcbtest()
{
	omp_lock_t	lock1, lock2;
	omp_nest_lock_t	lock3;

	ts_write("\n                  starting lockcbtest\n\n");

	// initialize the locks
	omp_init_lock(&lock1);
	omp_init_lock(&lock2);
	omp_init_nest_lock(&lock3);

#pragma omp parallel
	{
	    (*validate_ptr)("lockcb start");
#pragma omp master
	    {
	omp_set_lock(&lock1); 	// code pointer should be approximately label1
label1:	omp_unset_lock(&lock1);

	omp_set_lock(&lock2); 	// code pointer should be approximately label2
label2:	omp_unset_lock(&lock2);

	// now try a nested lock
	omp_set_nest_lock(&lock3);
	omp_set_nest_lock(&lock3);
	omp_set_nest_lock(&lock3);

	omp_unset_nest_lock(&lock3);
	omp_unset_nest_lock(&lock3);
	omp_unset_nest_lock(&lock3);
	    }

	    (*validate_ptr)("lockcb end");
	}
	omp_destroy_lock(&lock1);
	omp_destroy_lock(&lock2);
	omp_destroy_nest_lock(&lock3);
}

// ------------------------------------------------------------------------
// skew_delay -- burn CPU time to delay threads
// ------------------------------------------------------------------------
void
skew_delay(int count)
{
	int j,k;
	volatile float x;
	int	jmax;

	jmax = 7 * count;

	for ( j = 0; j < jmax; j++ ) {
	    x = 0;
	    for (k = 0; k < NSKEW; k ++ ) {
		x = x + 1.0;
	    }
	}
}

// ------------------------------------------------------------------------
// delay -- burn CPU time in main program to space out operations
// ------------------------------------------------------------------------
void
delay(int count)
{
	int j,k;
	volatile float x;
	int	jmax;

	jmax = 7 * count;

	for ( j = 0; j < jmax; j++ ) {
	    x = 0;
	    for (k = 0; k < NSKEW; k ++ ) {
		x = x + 1.0;
	    }
	}
}

#include "omptcb.h"
