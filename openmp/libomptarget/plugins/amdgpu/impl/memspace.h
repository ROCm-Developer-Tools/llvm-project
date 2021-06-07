#ifndef __MEMSPACE__H
#define __MEMSPACE__H

#include <cstdint>
#include <map>
#include <math.h>

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

class MemSpaceBase_  {
 public:
 MemSpaceBase_(uint64_t mem_size, uint64_t page_size) :
  mem_size(mem_size), page_size(page_size) {
    assert(mem_size % page_size == 0);
    num_pages = mem_size / page_size;
    log2page_size = log2l(page_size);
  }

  // give a \arg ptr calculates the table index of its containing page
  inline uint64_t calc_index(uintptr_t ptr) const {    
    return ptr >> log2page_size;
  }

protected:
  uint64_t mem_size;
  uint64_t page_size;
  uint64_t num_pages;
  // leading zero's for page size
  // used to calculate index in table
  uint64_t log2page_size;
};

class MemSpaceLinear_t : public MemSpaceBase_ {
 public:
 MemSpaceLinear_t(uint64_t mem_size, uint64_t page_size) :
  MemSpaceBase_(mem_size, page_size) {
    // init tab to zero
    tab = (uint8_t *) calloc(num_pages, sizeof(uint8_t));
  }

  // TODO: OpenMP will not remap same or subset region, only
  // completely separated regions. We can skip setting
  // table if the first element in the region is already set to 1
  void insert(const uintptr_t base, size_t size) {
    uint64_t start = calc_index(base);
    uint64_t end = calc_index(base+size-1);
    assert(start < num_pages);
    assert(end < num_pages);
    for(uint64_t i = start; i <= end; i++)
      tab[i] = 1;
  }

  // worst case complexity: O(n) with n = total number of pages
  // avg case complexity: O(num_pages) with num_pages = average number
  // of pages used by any allocation
  bool contains(const uintptr_t base, size_t size) const {
    uint64_t start = calc_index(base);
    uint64_t end = calc_index(base+size-1);
    for(uint64_t i = start; i <= end; i++)
      if(tab[i] == 0) return false;
    return true;
  }

  void dump() const {
    for(uint64_t i = 0; i < num_pages; i++)
      if(tab[i] != 0)
	printf("[%lu] = %d\n", i, tab[i]);
  }

 private:
  // the actual table that given a page index
  // contains whether the page belongs to the tracked
  // memory space
  // TODO: reduce to 1-bit per page and write access functions
  uint8_t *tab;
};

class MemSpaceLinearSmall_t : public MemSpaceBase_ {
 public:
 MemSpaceLinearSmall_t(uint64_t mem_size, uint64_t page_size) :
  MemSpaceBase_(mem_size, page_size) {
    // init tab to zero
    tab = (uint8_t *) calloc(num_pages, sizeof(uint8_t));
  }

  // TODO: OpenMP will not remap same or subset region, only
  // completely separated regions. We can skip setting
  // table if the first element in the region is already set to 1
  void insert(const uintptr_t base, size_t size) {
    uint64_t start = calc_index(base);
    uint64_t end = calc_index(base+size-1);
    assert(start < num_pages);
    assert(end < num_pages);
    for(uint64_t i = start; i <= end; i++)
      tab[i] = 1;
  }

  // worst case complexity: O(n) with n = total number of pages
  // avg case complexity: O(num_pages) with num_pages = average number
  // of pages used by any allocation
  bool contains(const uintptr_t base, size_t size) const {
    uint64_t start = calc_index(base);
    uint64_t end = calc_index(base+size-1);
    for(uint64_t i = start; i <= end; i++)
      if(tab[i] == 0) return false;
    return true;
  }

  void dump() const {
    for(uint64_t i = 0; i < num_pages; i++)
      if(tab[i] != 0)
	printf("[%lu] = %d\n", i, tab[i]);
  }

  // set the idx-th bit in tab_loc
  inline void set(uint64_t &tab_loc, const uint64_t idx) {
    // set idx-th bit in tab_loc
    tab_loc |= 1UL << idx;
  }
  
 private:
  // the actual table that given a page index
  // contains whether the page belongs to the tracked
  // memory space
  // implemented as a bit field
  uint64_t *tab;
};

#endif // __MEMSPACE__H
