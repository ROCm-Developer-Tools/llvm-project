#ifndef __MEMSPACE__H
#define __MEMSPACE__H

#include <cstdint>
#include <map>
#include <math.h>

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

class MemSpaceBase_ {
public:
  MemSpaceBase_(uint64_t mem_size, uint64_t page_size)
      : mem_size(mem_size), page_size(page_size) {
    assert(mem_size % page_size == 0);
    num_pages = mem_size / page_size;
    log2page_size = log2l(page_size);
  }

  // give a \arg ptr calculates the table index of its containing page
  inline uint64_t calc_page_index(uintptr_t ptr) const {
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
  MemSpaceLinear_t(uint64_t mem_size, uint64_t page_size)
      : MemSpaceBase_(mem_size, page_size) {
    // init tab to zero
    tab = (uint8_t *)calloc(num_pages, sizeof(uint8_t));
  }

  // TODO: OpenMP will not remap same or subset region, only
  // completely separated regions. We can skip setting
  // table if the first element in the region is already set to 1
  void insert(const uintptr_t base, size_t size) {
    uint64_t page_start = calc_page_index(base);
    uint64_t page_end = calc_page_index(base + size - 1);
    assert(page_start < num_pages);
    assert(page_end < num_pages);
    for (uint64_t i = page_start; i <= page_end; i++)
      tab[i] = 1;
  }

  // worst case complexity: O(n) with n = total number of pages
  // avg case complexity: O(num_pages) with num_pages = average number
  // of pages used by any allocation
  bool contains(const uintptr_t base, size_t size) const {
    uint64_t page_start = calc_page_index(base);
    uint64_t page_end = calc_page_index(base + size - 1);
    printf("Using 8bit/page version\n");
    for (uint64_t i = page_start; i <= page_end; i++)
      if (tab[i] == 0)
        return false;
    return true;
  }

  void dump() const {
    for (uint64_t i = 0; i < num_pages; i++)
      if (tab[i] != 0)
        printf("[%lu] = %d\n", i, tab[i]);
  }

private:
  // the actual table that given a page index
  // contains whether the page belongs to the tracked
  // memory space
  // TODO: reduce to 1-bit per page and write access functions
  uint8_t *tab;
};

// Same search semantics as Linear version, but uses a bit field
class MemSpaceLinearSmall_t : public MemSpaceBase_ {
public:
  MemSpaceLinearSmall_t(uint64_t mem_size, uint64_t page_size)
      : MemSpaceBase_(mem_size, page_size) {
    log2_pages_per_block = log2l(pages_per_block);
    assert((num_pages % 2) == 0);
    uint64_t tab_size = num_pages >> log2_pages_per_block;
    // init tab to zero
    tab = (uint64_t *)calloc(tab_size, sizeof(uint64_t));
  }

  virtual void insert(const uintptr_t base, size_t size) {
    uint64_t page_start = calc_page_index(base);
    uint64_t page_end = calc_page_index(base + size - 1);
    for (uint64_t i = page_start; i <= page_end; i++) {
      uint64_t blockId = i >> log2_pages_per_block;
      uint64_t blockOffset = i & (pages_per_block - 1);
      set(tab[blockId], blockOffset);
    }
  }

  // worst case complexity: O(n) with n = total number of pages
  // avg case complexity: O(num_pages) with num_pages = average number
  // of pages used by any allocation
  bool contains(const uintptr_t base, size_t size) const {
    uint64_t page_start = calc_page_index(base);
    uint64_t page_end = calc_page_index(base + size - 1);
    for (uint64_t i = page_start; i <= page_end; i++) {
      uint64_t blockId = i >> log2_pages_per_block;
      uint64_t blockOffset = i & (pages_per_block - 1);
      if (!isSet(tab[blockId], blockOffset))
        return false;
    }
    return true;
  }

  // set the idx-th bit in tab_loc
  inline void set(uint64_t &tab_loc, const uint64_t idx) {
    tab_loc |= 1UL << idx;
  }

  inline bool isSet(const uint64_t tab_loc, const uint64_t idx) const {
    return ((1UL << idx) == (tab_loc & (1UL << idx)));
  }

protected:
  // the actual table that given a page index
  // contains whether the page belongs to the tracked
  // memory space
  // implemented as a bit field
  uint64_t *tab;
  // this must be the same as the number of bits of each tab element
  const int pages_per_block = 64;
  int log2_pages_per_block;
};

// Same as Linear and Small, but based on OpenMP map clause
// restrictions: extension of mapped memory is not allowed,
// if the first bit of a coarse grain mapped page is set
// then all others will have been set as well. Cuts down
// on remapping of memory
class MemSpaceLinearSmallOMP_t : public MemSpaceLinearSmall_t {
public:
  MemSpaceLinearSmallOMP_t(uint64_t mem_size, uint64_t page_size)
      : MemSpaceLinearSmall_t(mem_size, page_size) {}
  bool test_and_insert(const uintptr_t base, size_t size) {
    uint64_t page_start = calc_page_index(base);
    uint64_t page_end = calc_page_index(base + size - 1);
    for (uint64_t i = page_start; i <= page_end; i++) {
      uint64_t blockId = i >> log2_pages_per_block;
      uint64_t blockOffset = i & (pages_per_block - 1);
      if (isSet(tab[blockId], blockOffset))
        return true;
      set(tab[blockId], blockOffset);
    }
    return false;
  }
};

#endif // __MEMSPACE__H
