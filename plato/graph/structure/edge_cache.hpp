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

// thread-safe edge cache implementation, fixed capacity
template <typename EDATA, typename VID_T = vid_t>
class edge_cache_t
  : public object_buffer_t<edge_unit_t<EDATA, VID_T>, mmap_allocator_t<edge_unit_t<EDATA, VID_T>>> {

public:
  using edata_t            = EDATA;
  using edge_unit_spec_t   = edge_unit_t<edata_t, VID_T>;

  explicit edge_cache_t(void) : object_buffer_t<edge_unit_t<EDATA, VID_T>, 
    mmap_allocator_t<edge_unit_t<EDATA, VID_T>>>() { }
  explicit edge_cache_t(size_t n) : object_buffer_t<edge_unit_t<EDATA, VID_T>, 
    mmap_allocator_t<edge_unit_t<EDATA, VID_T>>>(n) { }
};

// thread-safe edge block cache implementation, fixed capacity, can save memory
template <typename EDATA, typename VID_T = vid_t>
class edge_block_cache_t
  : public object_block_buffer_t<edge_unit_t<EDATA, VID_T>> {

public:
  using edata_t            = EDATA;
  using edge_unit_spec_t   = edge_unit_t<edata_t, VID_T>;

  explicit edge_block_cache_t(void) : object_block_buffer_t<edge_unit_t<EDATA, VID_T>>() { }
  explicit edge_block_cache_t(size_t block_num, size_t block_size) : 
    object_buffer_t<edge_unit_t<EDATA, VID_T>>(block_num, block_size) { }
};

// thread-safe edge cache implementation, fixed capacity
template <typename EDATA, typename VID_T = vid_t>
class edge_file_cache_t : public object_file_buffer_t<edge_unit_t<EDATA, VID_T>> {
public:
  using edata_t            = EDATA;
  using edge_unit_spec_t   = edge_unit_t<edata_t, VID_T>;

  explicit edge_file_cache_t(void) : object_file_buffer_t<edge_unit_t<EDATA, VID_T>>() { }
  explicit edge_file_cache_t(size_t n) : object_file_buffer_t<edge_unit_t<EDATA, VID_T>>(n) { }
};

}

