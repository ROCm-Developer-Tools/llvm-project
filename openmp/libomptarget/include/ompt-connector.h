//****************************************************************************
// global includes
//****************************************************************************

#include <dlfcn.h>
#include <string>



//****************************************************************************
// local includes
//****************************************************************************

#include <Debug.h>
#include <ompt.h>
#include <omptarget.h>



//****************************************************************************
// type declarations
//****************************************************************************

#define stringify(s) #s

#define LIBOMPTARGET_GET_TARGET_OPID libomptarget_get_target_opid



//****************************************************************************
// type declarations
//****************************************************************************

typedef void (*library_ompt_connect_t)(ompt_start_tool_result_t *result);

//----------------------------------------------------------------------------
// class library_ompt_connector_t
// purpose:
//
// establish connection between openmp runtime libraries
//
// NOTE: since this is used in attribute constructors, it should be declared
// within the constructor function to ensure that the class is initialized
// before it's methods are used
//----------------------------------------------------------------------------

class library_ompt_connector_t {
 public:

  void connect (ompt_start_tool_result_t *ompt_rtl_result) {
    initialize();
    if (library_ompt_connect) {
      library_ompt_connect(ompt_rtl_result);
    }
  };

  library_ompt_connector_t(const char *library_name) {
    library_connect_routine.append(library_name);
    library_connect_routine.append("_ompt_connect");
    is_initialized = false;
  };
  
 private:

  void initialize() {
    if (is_initialized == false) {
      DP("OMPT: library_ompt_connect = %s\n", library_connect_routine.c_str());
      void *vptr = dlsym(NULL, library_connect_routine.c_str());
      library_ompt_connect = reinterpret_cast<library_ompt_connect_t>
	(reinterpret_cast<long>(vptr));
      DP("OMPT: library_ompt_connect = %p\n", library_ompt_connect);
      is_initialized = true;
    }
  };

 private:

  bool is_initialized;
  library_ompt_connect_t library_ompt_connect; 
  std::string library_connect_routine;
};

