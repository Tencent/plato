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

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <sys/mman.h>

#include <vector>
#include <iostream>
#include <mutex>
#include <functional>
#include <algorithm>
#include <future>

#include "boost/intrusive/list.hpp"
#include "glog/logging.h"
#include "boost/format.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/iostreams/device/file.hpp"

namespace plato {

namespace thread_local_object_detail {

int create_object(std::function<void*()> construction, std::function<void(void *)> destruction);

void delete_object(int id);

[[gnu::hot]]
void* get_local_object(int id);

// thread safe.
unsigned objects_num();

// thread safe.
unsigned objects_num(int id);

// thread safe.
void object_foreach(int id, std::function<void(void*)> reducer);
}

struct thread_local_object_guard {
  int id_;
public:
  thread_local_object_guard(const thread_local_object_guard&) = delete;
  thread_local_object_guard& operator=(const thread_local_object_guard&) = delete;
  thread_local_object_guard(thread_local_object_guard&& x) noexcept : id_(x.id_) {
    x.id_ = -1;
  }
  thread_local_object_guard& operator=(thread_local_object_guard &&x) noexcept {
    if (this != &x) {
      this->~thread_local_object_guard();
      new(this) thread_local_object_guard(std::move(x));
    }
    return *this;
  }

  thread_local_object_guard(std::function<void*()> construction, std::function<void(void *)> destruction):
    id_(thread_local_object_detail::create_object(std::move(construction), std::move(destruction))) {
    if (-1 == id_) throw std::runtime_error("thread_local_object_detail::create_object failed.");
  }

  ~thread_local_object_guard() {
    if (id_ != -1) {
      thread_local_object_detail::delete_object(id_);
      id_ = -1;
    }
  }

  [[gnu::always_inline]] [[gnu::hot]]
  void* local() { return thread_local_object_detail::get_local_object(id_); }

  unsigned objects_num() { return thread_local_object_detail::objects_num(id_); }

  void foreach(std::function<void(void*)> reducer) { thread_local_object_detail::object_foreach(id_, std::move(reducer)); }
};

template <typename T>
class thread_local_object {
  thread_local_object_guard obj_;
public:
  thread_local_object() : obj_([] { return new T(); }, [] (void *p) { delete (T*)p; }) { }

  [[gnu::always_inline]] [[gnu::hot]]
  T* local() { return  (T*)(obj_.local()); }

  unsigned objects_num() { return obj_.objects_num(); }
};

class thread_local_buffer {
  thread_local_object_guard obj_;
public:
  thread_local_buffer(size_t capacity = (size_t)sysconf(_SC_PAGESIZE) * (size_t)sysconf(_SC_PHYS_PAGES) / 2) : obj_(
    [capacity] () {
      auto p = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
      CHECK(MAP_FAILED != p) << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno);
      return p;
    },
    [capacity] (void *p) {
      CHECK(-1 != munmap(p, capacity)) << boost::format("WARNING: munmap failed, err code: %d, err msg: %s") % errno % strerror(errno);
    }
  ) { }

  [[gnu::always_inline]] [[gnu::hot]]
  void* local() { return obj_.local(); }

  void foreach(std::function<void(void*)> reducer) { obj_.foreach(std::move(reducer)); }
};

class thread_local_counter {
  thread_local_object_guard obj_;
public:
  thread_local_counter() : obj_([] { return (void*)(new size_t(0)); }, [] (void* p) { delete (size_t*) p; }) { }

  [[gnu::always_inline]] [[gnu::hot]]
  size_t& local() { return *(size_t*)obj_.local(); }

  size_t reduce_sum() {
    size_t result = 0;
    obj_.foreach([&result] (void* p) {
      result += *(size_t*)p;
    });
    return result;
  }

  size_t reduce_max() {
    size_t result = 0;
    obj_.foreach([&result] (void* p) {
      result = std::max(result, *(size_t*)p);
    });
    return result;
  }

  size_t reduce_min() {
    size_t result = 0;
    obj_.foreach([&result] (void* p) {
      result = std::min(result, *(size_t*)p);
    });
    return result;
  }
};

}
