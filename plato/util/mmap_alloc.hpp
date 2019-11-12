/*
  Tencent is pleased to support the open source community by making
  Plato available.
  Copyright (C) 2019 THL A29 Limited, a Tencent company.
  All rights reserved.

  Licensed under the BSD 3-Clause License (the "License"); you may
  not use this file except in compliance with the License. You may
  obtain a copy of the License at

  https://opensource.org/licenses/BSD-3-Clause

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" basis,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
  implied. See the License for the specific language governing
  permissions and limitations under the License.

  See the AUTHORS file for names of contributors.
*/

#ifndef __PLATO_UTIL_MMAP_ALLOC_HPP__
#define __PLATO_UTIL_MMAP_ALLOC_HPP__

// debug
#include <cstdio>
// debug end

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cmath>
#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace plato {

namespace {

const int kPageSize = sysconf(_SC_PAGE_SIZE);

}

template <typename T>
class mmap_allocator_t {
public:
  using value_type      = T;
  using pointer         = T*;
  using const_pointer   = const T*;
  using reference       = T&;
  using const_reference = const T&;
  using size_type       = std::size_t;
  using difference_type = off_t;

  template <class U>
  friend class mmap_allocator_t;

  template <class U>
  struct rebind { typedef mmap_allocator_t<U> other; };

  using size_map_t = std::unordered_map<pointer, size_type>;

  mmap_allocator_t(void)
    : sizes_(new size_map_t()) { }

  template <class U>
  mmap_allocator_t(const mmap_allocator_t<U>& other)
    : sizes_(other.sizes_) { }

  pointer allocate(size_t n) {
    size_t to_alloc = n * sizeof(T);
    if (to_alloc % kPageSize != 0) {  // round up to multiple of page size
      to_alloc = ((to_alloc / kPageSize) + 1) * kPageSize;
    }

    pointer addr = static_cast<pointer>(mmap(nullptr, to_alloc,
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    assert(nullptr != addr);

    (*sizes_)[addr] = to_alloc;
    return addr;
  }

  void deallocate(pointer p, size_t /* n */) {
    assert(sizes_->count(p));
    munmap(p, (*sizes_)[p]);
  }

  void construct(pointer p, const_reference val) {
    new ((void*)p) T(val);
  }

  void destroy(pointer p) { p->~T(); }

  template <class U, class... Args>
  void construct(U* p, Args&&... args) {
    new ((void*)p) U(std::forward<Args>(args)...);
  }

  template <class U>
  void destroy(U* p) { p->~U(); }

  std::shared_ptr<size_map_t> sizes(void) { return sizes_; }

protected:
  std::shared_ptr<size_map_t> sizes_;
};

template <typename T, typename U>
inline bool operator==(const mmap_allocator_t<T>& a, const mmap_allocator_t<U>& b) {
  return a.sizes() == b.sizes();
}

template <typename T, typename U>
inline bool operator!=(const mmap_allocator_t<T>& a, const mmap_allocator_t<U>& b) {
  return !(a == b);
}

};  // namespace plato

#endif

