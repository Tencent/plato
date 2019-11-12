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

#ifndef __PLATO_UTIL_STREAM_HPP__
#define __PLATO_UTIL_STREAM_HPP__

#include <unistd.h>
#include <sys/mman.h>

#include <cmath>
#include <cstring>
#include <cstdint>

#include "plato/util/buffer.hpp"
#include "boost/iostreams/filtering_stream.hpp"

namespace plato {

struct mem_ostream_t {
  mem_ostream_t(const mem_ostream_t&) = delete;

  /**
   * @brief
   * @param reserved
   */
  mem_ostream_t(std::size_t reserved = 1024 * 20)
    : buf_(reserved),
    beg_(buf_.data_.get()),
    cur_(buf_.data_.get()),
    end_(buf_.data_.get() + buf_.size_)
  { }

  /**
   * @brief
   * @param ptr
   * @param size
   */
  mem_ostream_t(void* ptr, std::size_t size)
    : buf_(),
    beg_(static_cast<char*>(ptr)),
    cur_(static_cast<char*>(ptr)),
    end_(static_cast<char*>(ptr) + size)
  { }

  /**
   * @brief
   * @param other
   */
  mem_ostream_t(mem_ostream_t&& other)
    : buf_(std::move(other.buf_)),
      beg_(other.beg_),
      cur_(other.cur_),
      end_(other.end_) {
    other.beg_  = nullptr;
    other.cur_  = nullptr;
    other.end_  = nullptr;
  }

  /**
   * @brief
   */
  void reset(void) {
    beg_ = buf_.data_.get();
    cur_ = buf_.data_.get();
    end_ = buf_.data_.get() + buf_.size_;
  }

  /**
   * @brief getter
   * @return
   */
  size_t size(void) const { return (size_t)(cur_ - beg_); }

  /**
   * @brief append
   * @tparam T
   * @param tptr
   * @param size
   * @return
   */
  template<typename T>
  std::size_t write(const T* tptr, std::size_t size) {
    if (cur_ + size > end_) {
      shared_buffer_t::shared_array_type prev = buf_.data_;
      const std::size_t olds = static_cast<size_t>(cur_ - beg_);
      const std::size_t news = static_cast<size_t>(size + (olds * 2));

      buf_ = shared_buffer_t(news);
      std::memcpy(buf_.data_.get(), prev.get(), olds);

      beg_ = buf_.data_.get();
      cur_ = beg_ + olds;
      end_ = beg_ + news;
    }

    const std::uint8_t* ptr = (const uint8_t*)tptr;
    switch (size) {
      case 1 : std::memcpy(cur_, ptr, 1) ; break;
      case 2 : std::memcpy(cur_, ptr, 2) ; break;
      case 3 : std::memcpy(cur_, ptr, 3) ; break;
      case 4 : std::memcpy(cur_, ptr, 4) ; break;
      case 5 : std::memcpy(cur_, ptr, 5) ; break;
      case 6 : std::memcpy(cur_, ptr, 6) ; break;
      case 7 : std::memcpy(cur_, ptr, 7) ; break;
      case 8 : std::memcpy(cur_, ptr, 8) ; break;
      case 9 : std::memcpy(cur_, ptr, 9) ; break;
#if defined(__GNUC__) && defined(__SIZEOF_INT128__) // hack for detect int128 support
      case 16: std::memcpy(cur_, ptr, 16); break;
      case 17: std::memcpy(cur_, ptr, 17); break;
#endif
      default: std::memcpy(cur_, ptr, size);
    }
    cur_ += size;

    return size;
  }

  /**
   * @brief ge buffer
   * @return
   */
  shared_buffer_t get_buffer(void) const {  // copy a new buffer
    return shared_buffer_t(buf_.data_, static_cast<size_t>(cur_ - beg_));
  }

  /**
   * @brief
   * @return
   */
  intrusive_buffer_t get_intrusive_buffer(void) const {  // point to local buffer
    return intrusive_buffer_t(beg_, static_cast<size_t>(cur_ - beg_));
  }

protected:
  shared_buffer_t buf_;
  char* beg_;
  char* cur_;
  char* end_;
}; // struct mem_ostream_t

/***************************************************************************/

struct mem_istream_t {
  mem_istream_t(const mem_istream_t&) = delete;

  /**
   * @brief
   * @param ptr
   * @param size
   */
  mem_istream_t(const void* ptr, std::size_t size)
    : beg_(static_cast<const char*>(ptr)),
    cur_(static_cast<const char*>(ptr)),
    end_(static_cast<const char*>(ptr) + size)
  { }

  /**
   * @brief
   * @param buf
   */
  mem_istream_t(const intrusive_buffer_t& buf)
    : beg_(buf.data_),
    cur_(buf.data_),
    end_(buf.data_ + buf.size_)
  { }

  /**
   * @brief
   * @param buf
   */
  mem_istream_t(const shared_buffer_t &buf)
    : beg_(buf.data_.get()),
    cur_(buf.data_.get()),
    end_(buf.data_.get() + buf.size_)
  { }

  /**
   * @brief
   * @param other
   */
  mem_istream_t(mem_istream_t&& other)
    : beg_(other.beg_), cur_(other.cur_), end_(other.end_) {
    other.beg_ = nullptr;
    other.cur_ = nullptr;
    other.end_ = nullptr;
  }

  /**
   * @brief read some data
   * @tparam T
   * @param ptr
   * @param size
   * @return
   */
  template<typename T>
  std::size_t read(T* ptr, const std::size_t size) {
    const std::size_t avail = static_cast<size_t>(end_ - cur_);
    const std::size_t to_copy = (avail < size ? avail : size);
    switch (to_copy) {
      case 1 : std::memcpy(ptr, cur_, 1) ; break;
      case 2 : std::memcpy(ptr, cur_, 2) ; break;
      case 3 : std::memcpy(ptr, cur_, 3) ; break;
      case 4 : std::memcpy(ptr, cur_, 4) ; break;
      case 5 : std::memcpy(ptr, cur_, 5) ; break;
      case 6 : std::memcpy(ptr, cur_, 6) ; break;
      case 7 : std::memcpy(ptr, cur_, 7) ; break;
      case 8 : std::memcpy(ptr, cur_, 8) ; break;
      case 9 : std::memcpy(ptr, cur_, 9) ; break;
#if defined(__GNUC__) && defined(__SIZEOF_INT128__) // hack for detect int128 support
      case 16: std::memcpy(ptr, cur_, 16); break;
      case 17: std::memcpy(ptr, cur_, 17); break;
#endif
      default: std::memcpy(ptr, cur_, size);
    }
    cur_ += to_copy;

    return to_copy;
  }

  /**
   * @brief getter
   * @return
   */
  bool empty() const {
    return cur_ == end_;
  }
  /**
   * @brief getter
   * @return
   */
  char peekch() const {
    return *cur_;
  }
  /**
   * @brief getter and forward
   * @return
   */
  char getch() {
    return *cur_++;
  }
  /**
   * @brief
   */
  void ungetch(char) {
    --cur_;
  }

  /**
   * @brief
   * @return
   */
  shared_buffer_t get_buffer() const {
    return shared_buffer_t(cur_, static_cast<size_t>(end_ - cur_));
  }

  /**
   * @brief
   * @return
   */
  intrusive_buffer_t get_intrusive_buffer() const {
    return intrusive_buffer_t(cur_, static_cast<size_t>(end_ - cur_));
  }

protected:
  const char* beg_;
  const char* cur_;
  const char* end_;
}; // struct mem_istream_t

/***************************************************************************/

struct empty_ostream_t {
  size_t size_;

  empty_ostream_t() : size_(0) {}

  empty_ostream_t(const empty_ostream_t&) = delete;
  empty_ostream_t& operator=(const empty_ostream_t&) = delete;
  empty_ostream_t(empty_ostream_t&&) = delete;
  empty_ostream_t& operator=(empty_ostream_t&&) = delete;

  /**
   * @brief
   * @return
   */
  size_t size(void) const { return size_; }

  /**
   * @brief
   * @tparam T
   * @param size
   * @return
   */
  template<typename T>
  std::size_t write(const T*, std::size_t size) {
    size_ += size;
    return size;
  }
};

struct mem_simple_ostream_t {
  char* beg_;
  char* cur_;
  char* end_;

  mem_simple_ostream_t(char* beg, size_t capacity) :
    beg_(beg), cur_(beg), end_(beg + capacity) { }

  mem_simple_ostream_t(char* beg, char* end) :
    beg_(beg), cur_(beg), end_(end) { }

  mem_simple_ostream_t(const mem_simple_ostream_t&) = delete;
  mem_simple_ostream_t& operator=(const mem_simple_ostream_t&) = delete;
  mem_simple_ostream_t(mem_simple_ostream_t&&) = delete;
  mem_simple_ostream_t& operator=(mem_simple_ostream_t&&) = delete;

  /**
   * @brief
   * @return
   */
  size_t size(void) const { return cur_ - beg_; }

  /**
   * @brief
   * @tparam T
   * @param tptr
   * @param size
   * @return
   */
  template<typename T>
  std::size_t write(const T* tptr, std::size_t size) {
    auto* ptr = (const uint8_t*)tptr;
    switch (size) {
      case 1 : std::memcpy(cur_, ptr, 1) ; break;
      case 2 : std::memcpy(cur_, ptr, 2) ; break;
      case 3 : std::memcpy(cur_, ptr, 3) ; break;
      case 4 : std::memcpy(cur_, ptr, 4) ; break;
      case 5 : std::memcpy(cur_, ptr, 5) ; break;
      case 6 : std::memcpy(cur_, ptr, 6) ; break;
      case 7 : std::memcpy(cur_, ptr, 7) ; break;
      case 8 : std::memcpy(cur_, ptr, 8) ; break;
      case 9 : std::memcpy(cur_, ptr, 9) ; break;
#if defined(__GNUC__) && defined(__SIZEOF_INT128__) // hack for detect int128 support
      case 16: std::memcpy(cur_, ptr, 16); break;
      case 17: std::memcpy(cur_, ptr, 17); break;
#endif
      default: std::memcpy(cur_, ptr, size);
    }
    cur_ += size;

    return size;
  }
};

/***************************************************************************/

struct filtering_istream_t {
  filtering_istream_t(boost::iostreams::filtering_istream& is) : is_(is) { }

  /**
   * @brief
   * @tparam T
   * @param ptr
   * @param size
   * @return
   */
  template<typename T>
  std::size_t read(T* ptr, const std::size_t size) {
    is_.read((char*)ptr, size);
    return is_.good() ? size : 0;
  }

  /**
   * @brief
   * @return
   */
  bool empty() {
    is_.peek();
    return is_.eof();
  }

  /**
   * @brief
   * @return
   */
  char peekch() const {
    return is_.peek();
  }

  /**
   * @brief
   * @return
   */
  char getch() {
    return is_.get();
  }

  /**
   * @brief
   */
  void ungetch(char) {
    is_.unget();
  }
protected:
  boost::iostreams::filtering_istream& is_;
};

struct filtering_ostream_t {
  filtering_ostream_t(boost::iostreams::filtering_ostream& os) : os_(os) { }

  /**
   * @brief
   * @tparam T
   * @param ptr
   * @param size
   * @return
   */
  template<typename T>
  std::size_t write(const T* ptr, std::size_t size) {
    os_.write((char*)ptr, size);
    return os_.good() ? size : 0;
  }
protected:
  boost::iostreams::filtering_ostream& os_;
};

}  // namespace plato

#endif

