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

#ifndef __PLATO_BUFFERS_HPP__
#define __PLATO_BUFFERS_HPP__

#include <unistd.h>

#include <cstring>
#include <memory>
#include <type_traits>

namespace plato {

struct intrusive_buffer_t {
  intrusive_buffer_t(const char* data, std::size_t size)
    : data_(data), size_(size) { }

  intrusive_buffer_t(const intrusive_buffer_t& o)
    : data_(o.data_), size_(o.size_) { }

  const char*       data_;
  const std::size_t size_;

protected:
  intrusive_buffer_t();
};

/***************************************************************************/

struct shared_buffer_t {
  typedef std::shared_ptr<char> shared_array_type;

  explicit shared_buffer_t(std::size_t size = 0)
    :size_(0)
  { resize(size); }

  shared_buffer_t(const void* ptr, std::size_t size)
    :size_(0)
  { assign(ptr, size); }

  shared_buffer_t(shared_array_type buf, std::size_t size)
    :size_(size)
  { if (size) { data_ = std::move(buf); } }

  shared_buffer_t(const shared_buffer_t& buf)
    :size_(buf.size_)
  { if (size_) { data_ = buf.data_; } }

  shared_buffer_t(shared_buffer_t&& buf)
    :data_(std::move(buf.data_)),
    size_(buf.size_)
  { buf.size_ = 0; }

  shared_buffer_t& operator=(const shared_buffer_t&) = default;
  shared_buffer_t& operator=(shared_buffer_t&&) = default;

  void resize(std::size_t new_size) {
    if (new_size > size_) {
      data_.reset(new char[new_size], &deleter);
    }
    size_ = new_size;
  }

  void assign(const void* ptr, std::size_t size) {
    resize(size);
    if (size_) {
      std::memcpy(data_.get(), ptr, size_);
    }
  }

  shared_array_type data_;
  std::size_t       size_;

protected:
  static void deleter(char* ptr) { delete[] ptr; }
};

/***************************************************************************/

}  // namespace plato

#endif // buffers.hpp

