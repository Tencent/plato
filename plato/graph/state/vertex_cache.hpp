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

#include "plato/util/object_buffer.hpp"
#include "plato/util/mmap_alloc.hpp"

namespace plato {

// thread-safe vertex cache implementation, fixed capacity
template <typename VDATA>
class vertex_cache_t
  : public object_buffer_t<vertex_unit_t<VDATA>, mmap_allocator_t<vertex_unit_t<VDATA>>> {

public:
  using vdata_t            = VDATA;
  using vertex_unit_spec_t = vertex_unit_t<vdata_t>;

  explicit vertex_cache_t(void) : object_buffer_t<vertex_unit_t<VDATA>, mmap_allocator_t<vertex_unit_t<VDATA>>>() { }
  explicit vertex_cache_t(size_t n) : object_buffer_t<vertex_unit_t<VDATA>, mmap_allocator_t<vertex_unit_t<VDATA>>>(n) { }
};

// thread-safe vertex cache implementation, fixed capacity
template <typename VDATA>
class vertex_file_cache_t
  : public object_file_buffer_t<vertex_unit_t<VDATA>, mmap_allocator_t<vertex_unit_t<VDATA>>> {

public:
  using vdata_t            = VDATA;
  using vertex_unit_spec_t = vertex_unit_t<vdata_t>;

  explicit vertex_file_cache_t(void) : object_file_buffer_t<vertex_unit_t<VDATA>, mmap_allocator_t<vertex_unit_t<VDATA>>>() { }
  explicit vertex_file_cache_t(size_t n) : object_file_buffer_t<vertex_unit_t<VDATA>, mmap_allocator_t<vertex_unit_t<VDATA>>>(n) { }
};

}

