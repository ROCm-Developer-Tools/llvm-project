/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef SRC_RUNTIME_INCLUDE_RT_H_
#define SRC_RUNTIME_INCLUDE_RT_H_

#include "atmi_runtime.h"
#include "hsa.h"
#include <cstdarg>
#include <string>
#include <set>

#include <iostream>

using namespace std;

namespace core {

#define DEFAULT_MAX_QUEUE_SIZE 4096
#define DEFAULT_DEBUG_MODE 0
class Environment {
public:
  Environment()
      : max_queue_size_(DEFAULT_MAX_QUEUE_SIZE),
        debug_mode_(DEFAULT_DEBUG_MODE) {
    GetEnvAll();
  }

  void GetEnvAll();

  int getMaxQueueSize() const { return max_queue_size_; }
  int getDebugMode() const { return debug_mode_; }

private:
  std::string GetEnv(const char *name) {
    char *env = getenv(name);
    std::string ret;
    if (env) {
      ret = env;
    }
    return ret;
  }

  int max_queue_size_;
  int debug_mode_;
};

 
class Runtime final {
public:
  static Runtime &getInstance() {
    static Runtime instance;
    return instance;
  }

  // init/finalize
  static atmi_status_t Initialize();
  static atmi_status_t Finalize();
  // machine info
  static atmi_machine_t *GetMachineInfo();
  // modules
  static atmi_status_t RegisterModuleFromMemory(
      void *, size_t, atmi_place_t,
      atmi_status_t (*on_deserialized_data)(void *data, size_t size,
                                            void *cb_state),
      void *cb_state);

  // data
  static atmi_status_t Memcpy(hsa_signal_t, void *, const void *, size_t);
  static atmi_status_t Memfree(void *);
  static atmi_status_t Malloc(void **, size_t, atmi_mem_place_t);

  int getMaxQueueSize() const { return env_.getMaxQueueSize(); }
  int getDebugMode() const { return env_.getDebugMode(); }

protected:
  Runtime() = default;
  ~Runtime() = default;
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

protected:
  // variable to track environment variables
  Environment env_;
};

 // TODO: extract from memory pool
static const uintptr_t pageSize = 4096;
 
typedef uintptr_t CoarseGrainHstPtr;

class CoarseGrainElemTy final {
 public:
   CoarseGrainElemTy(CoarseGrainHstPtr begin, CoarseGrainHstPtr end) :
     begin_(begin), end_(end) {}

   static int pageAlignPrev(CoarseGrainHstPtr ptr) {
     return (ptr & ~(pageSize-1));
   }
   static int pageAlignNext(CoarseGrainHstPtr ptr) {
     auto pad = (pageSize - (ptr%pageSize))%pageSize;
     return (ptr + pad);
   }
   CoarseGrainHstPtr getBegin() const { return begin_; }
   CoarseGrainHstPtr getEnd() const { return end_; }

 private:
   CoarseGrainHstPtr begin_;
   CoarseGrainHstPtr end_;
 };

inline bool operator<(const CoarseGrainElemTy &lhs, const CoarseGrainElemTy &rhs) {
  return lhs.getBegin() < rhs.getBegin();
}

inline bool operator<(const CoarseGrainElemTy &lhs, const CoarseGrainHstPtr &rhs) {
  return lhs.getBegin() < rhs;
}

inline bool operator<(const CoarseGrainHstPtr &lhs, const CoarseGrainElemTy &rhs) {
  return lhs < rhs.getBegin();
}

typedef std::set<CoarseGrainElemTy, std::less<>> CoarseGrainTableTy;

// Table of host memory areas that are marked as coarse grain
// static beacuse it is shared between different devices and their RTLs
// TODO: access via locks
static CoarseGrainTableTy CoarseGrainMemTable_;

static int IsCoarseGrain(CoarseGrainHstPtr begin, size_t size) {
  if (CoarseGrainMemTable_.empty()) return 0;
  
  auto upper = CoarseGrainMemTable_.upper_bound(begin);
  if(upper == CoarseGrainMemTable_.end()) {
    // begin is beyond start address of last entry, could be included in it
    auto prev = std::prev(upper);
    if (begin+(CoarseGrainHstPtr)size-1 < prev->getEnd())
      return 1;

    // goes beyond last entry
    return 0;
  }

  // upper exists

  // upper is only element in the set
  if (upper == CoarseGrainMemTable_.begin()) {
    if(begin == upper->getBegin() && ((begin+(CoarseGrainHstPtr)size-1 < upper->getEnd())))
      return 1;
    // begin,begin+size-1 does not fit in first and only element in the table
    return 0;				       
  }

  // upper exists and it is not the first element
  cout << "IsCoarseGrain: upper begin = " << upper->getBegin() << " end = " << upper->getEnd() << "\n";
  if (begin < upper->getBegin()) {
    // begin is before first mapped region: at least one element is outside
    if (upper != CoarseGrainMemTable_.begin())
      return 0;
      
    // either begin falls in the previous area or in the middle between the previous and upper      
    auto prev = std::prev(upper);
    cout << "IsCoarseGrain: prev begin = " << prev->getBegin() << " end = " << prev->getEnd() << "\n";
    if (begin+(CoarseGrainHstPtr)size-1 < prev->getEnd())
      return 1;
    return 0;
  }
    
  // begin corresponds to begin of upper, check if it is inside or extends after
  if (begin+(CoarseGrainHstPtr)size-1 < upper->getEnd())
    return 1;
  return 0;
}
} // namespace core


#endif // SRC_RUNTIME_INCLUDE_RT_H_
