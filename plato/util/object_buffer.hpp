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

#include <unistd.h>
#include <sys/mman.h>

#include <cstring>
#include <memory>
#include <atomic>
#include <type_traits>
#include <mutex>
#include <condition_variable>

#include "yas/binary_oarchive.hpp"
#include "yas/binary_iarchive.hpp"
#include "boost/align.hpp"
#include "glog/logging.h"

#include "plato/graph/base.hpp"
#include "mmap_alloc.hpp"
#include "atomic.hpp"
#include "spinlock.hpp"
#include "perf.hpp"
#include "temporary_file.hpp"
#include "background_executor.hpp"
#include "stream.hpp"
#include "foutput.h"

namespace plato {

struct object_buffer_opt_t {
  size_t capacity_ = 0;
  std::string prefix_;
  std::string path_ = ".cache";
};

// fixed-size, object buffer with thread-safe traversal
template <typename T, typename ALLOC = std::allocator<T>>
class object_buffer_t {
public:

  // ******************************************************************************* //
  // required types & methods

  using object_t       = T;
  using allocator_type = ALLOC;

  /**
   * @brief getter
   * @return
   */
  size_t size(void) { return size_; }

  /**
   * @brief getter
   * @return
   */
  size_t capacity(void) { return capacity_; }

  /**
   * @brief destroy all objects in buffer and set size to 0
   */
  void   clear(void);

  /**
   * @brief thread-safe emplace_back
   * @tparam Args
   * @param args
   * @return
   */
  template <typename... Args>
  size_t emplace_back(Args&&... args);

  /**
   * @brief thread-safe push_back
   * @param item
   * @return
   */
  size_t push_back(const T& item);

  /**
   * @brief thread-safe push_back
   * @param pitems
   * @param n
   * @return
   */
  size_t push_back(const T* pitems, size_t n);

  /**
   * @brief not thread-safe access methods
   * @param i
   * @return
   */
  object_t& operator[] (size_t i);

  /**
   * @brief not thread-safe access methods
   * @param i
   * @return
   */
  const object_t& operator[] (size_t i) const;

  // traversal related

  /**
   * @brief not thread-safe access methods
   * @param opts
   */
  void reset_traversal(const traverse_opts_t& opts = traverse_opts_t());

  /*
   * process a chunk of objects, thread-safe
   *
   * \param traversal    process callback functor, shoud implement
   *                     <tt> void operator()(size_t idx, T* object) <\tt>
   * \param chunk_size   input, traverse at most chunk_size objects,
   *                     output, real objects processed
   *
   * \return
   *    true  -- traverse at lease one object
   *    false -- no more object to traverse
   * */
  template <typename Traversal>
  bool next_chunk(Traversal&& traversal, size_t* chunk_size);

  // ********************* end of required types & methods ************************** //

  /**
   * @brief set capacity of the buffer to half of the system's memory
   */
  object_buffer_t(void);

  /**
   * @brief constructor with capacity
   * @param capacity
   * @param alloc
   */
  object_buffer_t(size_t capacity, const ALLOC& alloc = ALLOC());

  /**
   * @brief constructor with option
   * @param opt
   */
  object_buffer_t(object_buffer_opt_t opt);

  object_buffer_t& operator=(object_buffer_t&& x);
  object_buffer_t(object_buffer_t&& other);

  object_buffer_t(const object_buffer_t&) = delete;
  object_buffer_t& operator=(const object_buffer_t&) = delete;

  /**
   * @brief destructor
   */
  ~object_buffer_t(void);

  constexpr static bool is_trivial(void) { return std::is_trivial<T>::value; }

protected:
  size_t               capacity_;
  size_t               size_;
  object_t*            objs_;
  allocator_type       allocator_;

  traverse_opts_t                        traverse_opts_;
  std::atomic<size_t>                    traverse_i_;
  std::vector<std::pair<size_t, size_t>> traverse_range_;
};

// ******************************************************************************* //
// implementations

template <typename T, typename ALLOC>
object_buffer_t<T, ALLOC>::object_buffer_t(void)
    : size_(0), objs_(nullptr), allocator_(ALLOC()), traverse_i_(0) {
  capacity_ = (size_t)sysconf(_SC_PAGESIZE) * (size_t)sysconf(_SC_PHYS_PAGES) / sizeof(T) / 2;
  objs_ = allocator_.allocate(capacity_);
  CHECK(nullptr != objs_);
}

template <typename T, typename ALLOC>
object_buffer_t<T, ALLOC>::object_buffer_t(size_t capacity, const ALLOC& alloc)
    : capacity_(capacity), size_(0), objs_(nullptr), allocator_(alloc),
      traverse_i_(0) {
  objs_ = allocator_.allocate(capacity_);
  CHECK(nullptr != objs_);
}

template <typename T, typename ALLOC>
object_buffer_t<T, ALLOC>::object_buffer_t(object_buffer_opt_t opt) :
    size_(0), objs_(nullptr), allocator_(ALLOC()), traverse_i_(0) {
  if (!opt.capacity_) {
    opt.capacity_ = (size_t)sysconf(_SC_PAGESIZE) * (size_t)sysconf(_SC_PHYS_PAGES) / sizeof(T) / 2;
  }
  capacity_ = opt.capacity_;
  objs_ = allocator_.allocate(capacity_);
  CHECK(nullptr != objs_);
}

template <typename T, typename ALLOC>
object_buffer_t<T, ALLOC>::~object_buffer_t(void) {
  clear();
  if (objs_) {
    allocator_.deallocate(objs_, capacity_);
  }
}

template <typename T, typename ALLOC>
object_buffer_t<T, ALLOC>& object_buffer_t<T, ALLOC>::operator=(object_buffer_t<T, ALLOC>&& x) {
  capacity_  = x.capacity_;
  size_      = x.size_;
  objs_      = x.objs_;
  allocator_ = x.allocator_;
  traverse_i_.store(0);
  traverse_range_.clear();

  x.objs_     = nullptr;
  x.size_     = 0;
  x.capacity_ = 0;
  x.traverse_range_.clear();

  return *this;
}

template <typename T, typename ALLOC>
object_buffer_t<T, ALLOC>::object_buffer_t(object_buffer_t<T, ALLOC>&& other) {
  this->operator=(std::forward<object_buffer_t>(other));
}

template <typename T, typename ALLOC>
typename object_buffer_t<T, ALLOC>::object_t& object_buffer_t<T, ALLOC>::operator[] (size_t i) {
  CHECK(i < size_) << i << " exceed size_: " << size_;
  return objs_[i];
}

template <typename T, typename ALLOC>
const typename object_buffer_t<T, ALLOC>::object_t& object_buffer_t<T, ALLOC>::operator[] (size_t i) const {
  CHECK(i < size_) << i << " exceed size_: " << size_;
  return objs_[i];
}

template <typename T, typename ALLOC >
void object_buffer_t<T, ALLOC>::clear(void) {
  if (false == std::is_trivial<object_t>::value) {
    for (size_t i = 0; i < size_; ++i) {
      allocator_.destroy(&objs_[i]);
    }
  }
  size_ = 0;
}

template <typename T, typename ALLOC >
template <typename... Args>
size_t object_buffer_t<T, ALLOC>::emplace_back(Args&&... args) {
  size_t idx = __sync_fetch_and_add(&size_, 1);
  CHECK(idx < capacity_) << "object buffer overflow, idx: " << idx << ", capacity_: " << capacity_;
  allocator_.construct(&objs_[idx], std::forward<Args>(args)...);
  return idx;
}

template <typename T, typename ALLOC>
size_t object_buffer_t<T, ALLOC>::push_back(const T& item) {
  size_t idx = __sync_fetch_and_add(&size_, 1);
  CHECK(idx < capacity_) << "object buffer overflow, idx: " << idx << ", capacity_: " << capacity_;
  allocator_.construct(&objs_[idx], item);
  return idx;
}

template <typename T, typename ALLOC>
size_t object_buffer_t<T, ALLOC>::push_back(const T* item, size_t n) {
  size_t idx = __sync_fetch_and_add(&size_, n);
  CHECK(idx + n <= capacity_) << "object buffer overflow, idx: " << idx << ", n: " << n << ", capacity_: " << capacity_;

  for (size_t i = 0; i < n; ++i) {
    allocator_.construct(&objs_[idx + i], item[i]);
  }
  return idx;
}

template <typename T, typename ALLOC>
void object_buffer_t<T, ALLOC>::reset_traversal(const traverse_opts_t& opts) {
  traverse_i_.store(0, std::memory_order_relaxed);
  traverse_opts_ = opts;

  size_t basic_chunk = PAGESIZE;
  auto build_ranges_origin = [&](void) {
    size_t buckets = std::min((size_t)(MBYTES), size_ / basic_chunk + 1);

    traverse_range_.resize(buckets);
    size_t remained_items = size_;
    size_t o_start = 0;

    for (size_t i = 0; i < buckets; ++i) {
      if ((buckets - 1) == i) {
        traverse_range_[i].first  = o_start;
        traverse_range_[i].second = size_;
      } else {
        size_t amount = remained_items / (buckets - i) / basic_chunk * basic_chunk;

        traverse_range_[i].first  = o_start;
        traverse_range_[i].second = o_start + amount;
      }

      o_start = traverse_range_[i].second;
      remained_items -= (traverse_range_[i].second - traverse_range_[i].first);
    }
    CHECK(0 == remained_items);
  };

  switch (traverse_opts_.mode_) {
  case traverse_mode_t::ORIGIN:
    build_ranges_origin();
    break;
  case traverse_mode_t::RANDOM:
    build_ranges_origin();
    std::random_shuffle(traverse_range_.begin(), traverse_range_.end());
    break;
  default:
    CHECK(false) << "unknown/unsupported traverse mode!";
  }

#ifdef __BUFFER_DEBUG__
  if (0 == cluster_info_t::get_instance().partition_id_) {
    for (size_t i = 0; i < traverse_range_.size(); ++i) {
      LOG(INFO) << "traverse_range_[" << i << "]: " << traverse_range_[i].first << ", " << traverse_range_[i].second;
    }
  }
#endif
}

template <typename T, typename ALLOC>
template <typename Traversal>
bool object_buffer_t<T, ALLOC>::next_chunk(Traversal&& traversal, size_t* chunk_size) {
  size_t range_start = traverse_i_.fetch_add(*chunk_size, std::memory_order_relaxed);

  if (range_start >= traverse_range_.size()) { return false; }
  if (range_start + *chunk_size > traverse_range_.size()) {
    *chunk_size = traverse_range_.size() - range_start;
  }

  size_t range_end = range_start + *chunk_size;
  for (size_t range_i = range_start; range_i < range_end; ++range_i) {
    size_t o_start = traverse_range_[range_i].first;
    size_t o_end   = traverse_range_[range_i].second;

    for (size_t o_i = o_start; o_i < o_end; ++o_i) {
      CHECK(o_i < capacity_);
      traversal(o_i, &objs_[o_i]);
    }
  }

  return true;
}

// fixed-size, object block buffer with thread-safe traversal
template <typename T>
class object_block_buffer_t {
public:
  using allocator_type = mmap_allocator_t<T>;

  /**
   * @brief constructor
   * @param block_num_
   * @param block_size
   */
  object_block_buffer_t(size_t block_num_ = 1024, size_t block_size = HUGESIZE);

  /**
   * @brief getter
   * @return
   */
  size_t size(void) { return size_; }

  /**
   * @brief getter
   * @param need_block_num
   * @return
   */
  size_t get_appropriate_block_num(size_t need_block_num);

  /**
   * @brief
   * @param new_block_num
   */
  void expand_blocks(size_t new_block_num);

  /**
   * @brief thread-safe
   * @param item
   * @return
   */
  size_t push_back(const T& item);

  /**
   * @brief thread-safe
   * @param pitems
   * @param n
   * @return
   */
  size_t push_back(const T* pitems, size_t n);

  /**
   * @brief not thread-safe access methods
   * @param i
   * @return
   */
  T& operator[] (size_t i);

  /**
   * @brief not thread-safe access methods
   * @param i
   * @return
   */
  const T& operator[] (size_t i) const;

  /**
   * @brief reset and traversal from begin.
   * @param opts
   */
  void reset_traversal(const traverse_opts_t& opts = traverse_opts_t());

  template <typename Traversal>
  bool next_chunk(Traversal&& traversal, size_t* chunk_size);
private:
  size_t block_num_;
  size_t size_;
  size_t block_size_;
  std::atomic<size_t> traverse_i_;
  std::vector<std::shared_ptr<object_buffer_t<T, allocator_type>>> buffers_;
  std::mutex mutex_;
  traverse_opts_t opts_;
  size_t used_block_num_;
};

template <typename T>
object_block_buffer_t<T>::object_block_buffer_t(size_t block_num, size_t block_size) :
    block_num_(block_num), size_(0), block_size_(block_size), traverse_i_(0)  {
  size_t max_item_num = (size_t)sysconf(_SC_PAGESIZE) * (size_t)sysconf(_SC_PHYS_PAGES) / sizeof(T);
  size_t max_block_num = (max_item_num + block_size - 1) / block_size;
  buffers_.reserve(max_block_num); // make sure there is no need to expand memory
  for (size_t i = 0; i < block_num; ++i) { //just init first block_num buffers
    std::shared_ptr<object_buffer_t<T, allocator_type>> buffer_ptr(new object_buffer_t<T, allocator_type>(block_size));
    buffers_.push_back(buffer_ptr);
  }
}

template <typename T>
size_t object_block_buffer_t<T>::get_appropriate_block_num(size_t need_block_num) {
  size_t num = block_num_;
  while(num < need_block_num) {
    num *= 2;
  }
  return num;
}

template <typename T>
void object_block_buffer_t<T>::expand_blocks(size_t need_block_num) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (need_block_num > block_num_) {
    size_t expand_num = need_block_num - block_num_;
    for (size_t i = 0; i < expand_num; ++i) {
      std::shared_ptr<object_buffer_t<T, allocator_type>> buffer_ptr(new object_buffer_t<T, allocator_type>(block_size_));
      buffers_.push_back(buffer_ptr);
    }
    block_num_ = need_block_num;
  }
}

template <typename T>
size_t object_block_buffer_t<T>::push_back(const T& item) {
  size_t idx = __sync_fetch_and_add(&size_, (size_t)1);
  size_t need_block_num = (idx + 1 + block_size_ - 1) / block_size_;
  if (need_block_num > block_num_) {
    expand_blocks(get_appropriate_block_num(need_block_num));
  }
  size_t block_id = idx / block_size_;
  buffers_[block_id]->push_back(item);
  return idx;
}

template <typename T>
size_t object_block_buffer_t<T>::push_back(const T* pitems, size_t n) {
  size_t idx = __sync_fetch_and_add(&size_, n);
  size_t need_block_num = (idx + n + block_size_ - 1) / block_size_;
  if (need_block_num > block_num_) {
    expand_blocks(get_appropriate_block_num(need_block_num));
  }

  size_t block_id = idx / block_size_;
  const T* item_ptr = pitems;
  while (block_id < block_num_) {
    if (block_id * block_size_ >= idx + n) break;
    size_t left = std::max(idx, block_id * block_size_);
    size_t right = std::min(idx + n, (block_id + 1) * block_size_);
    buffers_[block_id]->push_back(item_ptr, right - left);
    item_ptr += right - left;
    block_id++;
  }

  return idx;
}

template <typename T>
T& object_block_buffer_t<T>::operator[] (size_t i) {
  CHECK(i < size_) << i << " exceed size_: " << size_;
  return (*(buffers_[i / block_size_]))[i % block_size_];
}

template <typename T>
const T& object_block_buffer_t<T>::operator[] (size_t i) const {
  CHECK(i < size_) << i << " exceed size_: " << size_;
  return (*(buffers_[i / block_size_]))[i % block_size_];
}


template <typename T>
void object_block_buffer_t<T>::reset_traversal(const traverse_opts_t& opts) {
  opts_ = opts;
  traverse_i_.store(0, std::memory_order_relaxed);
  used_block_num_ = (size_ + block_size_ - 1) / block_size_;
}

template <typename T>
template <typename Traversal>
bool object_block_buffer_t<T>::next_chunk(Traversal&& traversal, size_t* chunk_size) {
  size_t max_chunk_size = used_block_num_ / omp_get_num_threads() + 1;
  if (max_chunk_size < *chunk_size) *chunk_size = max_chunk_size;
  size_t range_start = traverse_i_.fetch_add(*chunk_size, std::memory_order_relaxed);
  if (range_start >= used_block_num_ ) return false;
  if (range_start + *chunk_size >= used_block_num_) {
    *chunk_size = used_block_num_ - range_start;
  }
  size_t range_end = range_start + *chunk_size;

  for (size_t i = range_start; i < range_end; ++i) {
    size_t block_size = buffers_[i]->size();
    if (block_size == 0) continue;
    T* ptr = &((*buffers_[i])[0]);
    for (size_t j = 0; j < block_size; ++j) {
      traversal(j, ptr + j);
    }

    if (opts_.auto_release_) {
      buffers_[i].reset();
    }
  }

  return true;
}

// ******************************************************************************* //

namespace object_buffer_detail {

enum prefetch_status {
  need_prefetch,
  prefetching,
  prefetched,
};

template <typename T>
struct traverse_unit {
  int fd_ = -1;
  size_t offset_begin_ = 0;
  size_t offset_end_ = 0;
  void* map_begin_ = nullptr;
  T* objs_begin_ = nullptr;
  T* objs_end_ = nullptr;
  std::mutex mutex_;
  std::condition_variable cv_;
  prefetch_status status_ = prefetch_status::need_prefetch;

  traverse_unit(const traverse_unit&) = delete;
  traverse_unit& operator=(const traverse_unit&) = delete;
  traverse_unit(traverse_unit&& x) noexcept :
      fd_(x.fd_), offset_begin_(x.offset_begin_),
      offset_end_(x.offset_end_), map_begin_(x.map_begin_),
      objs_begin_(x.objs_begin_), objs_end_(x.objs_end_), status_(x.status_) {
    x.fd_ = -1;
    x.offset_begin_ = 0;
    x.map_begin_ = nullptr;
    x.objs_begin_ = nullptr;
    x.objs_end_ = nullptr;
  }
  traverse_unit& operator=(traverse_unit&& x) noexcept {
    if (this != &x) {
      this->~traverse_unit();
      new (this) traverse_unit(std::move(x));
    }
    return *this;
  }

  traverse_unit(int fd, size_t offset_begin, size_t offset_end) :
      fd_(fd), offset_begin_(offset_begin), offset_end_(offset_end) { }

  ~traverse_unit() { evict(); }

  bool prefetch(bool wait) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      switch (status_) {
      case prefetch_status::need_prefetch:
        status_ = prefetch_status::prefetching;
        break;
      case prefetch_status::prefetching:
        if(wait) cv_.wait(lock);
        return false;
      case prefetch_status::prefetched:
        return false;
      default:
        abort();
      }
    }

    CHECK(fd_ != -1 && offset_begin_ < offset_end_ && !map_begin_ && !objs_begin_ && !objs_end_);
    size_t file_offset_begin = boost::alignment::align_down(sizeof(T) * offset_begin_, PAGESIZE);
    size_t file_offset_end = sizeof(T) * offset_end_;
    map_begin_ = mmap(0, file_offset_end - file_offset_begin, PROT_READ, MAP_PRIVATE | MAP_POPULATE | MAP_NORESERVE, fd_, file_offset_begin);
    CHECK(MAP_FAILED != map_begin_)
    << boost::format("WARNING: mmap failed, err code: %d, err msg: %s.") % errno % strerror(errno);
    objs_begin_ = (T*)((char*)map_begin_ + sizeof(T) * offset_begin_ - file_offset_begin);
    objs_end_ = objs_begin_ + offset_end_ - offset_begin_;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      status_ = prefetch_status::prefetched;
      cv_.notify_all();
    }
    return true;
  }

  void evict() {
    if (map_begin_) {
      CHECK(fd_ != -1 && offset_begin_ < offset_end_ && objs_begin_ && objs_end_);
      auto p = map_begin_;
      map_begin_ = nullptr;
      objs_begin_ = nullptr;
      objs_end_ = nullptr;

      size_t file_offset_begin = boost::alignment::align_down(sizeof(T) * offset_begin_, PAGESIZE);
      size_t file_offset_end = sizeof(T) * offset_end_;
      CHECK(-1 != munmap(p, file_offset_end - file_offset_begin))
      << boost::format("WARNING: munmap failed, err code: %d, err msg: %s") % errno % strerror(errno);
    }
  }
};

}

// ******************************************************************************* //

// fixed-size, object file buffer with thread-safe traversal
template <typename T, typename Enable = void>
class object_file_buffer_t {
  constexpr static size_t mmap_unit_capacity = 32 * MBYTES;
public:
  size_t size(void) { return size_; }
  size_t capacity(void) { return capacity_; }

  // thread-safe push_back
  size_t push_back(const T& item) { return push_back(&item, 1); }

  // thread-safe push_back
  size_t push_back(const T* pitems, size_t n);

  // traversal related

  // not thread-safe
  void reset_traversal(const traverse_opts_t& = traverse_opts_t());

  /*
   * process a chunk of objects, thread-safe
   *
   * \param traversal    process callback functor, shoud implement
   *                     <tt> void operator()(size_t idx, T* object) <\tt>
   * \param chunk_size   input, traverse at most chunk_size objects,
   *                     output, real objects processed
   *
   * \return
   *    true  -- traverse at lease one object
   *    false -- no more object to traverse
   * */
  template <typename Traversal>
  bool next_chunk(Traversal&& traversal, size_t* chunk_size);

  // ********************* end of required types & methods ************************** //
  object_file_buffer_t(
      size_t capacity = 0,
      std::string cache_dir = ".cache/");

  object_file_buffer_t(object_buffer_opt_t opt);

  object_file_buffer_t(object_file_buffer_t&& other) noexcept;
  object_file_buffer_t& operator=(object_file_buffer_t&& x) noexcept;

  object_file_buffer_t(const object_file_buffer_t&) = delete;
  object_file_buffer_t& operator=(const object_file_buffer_t&) = delete;

  constexpr static bool is_trivial() { return std::is_trivial<T>::value; }
protected:
  size_t                                                  capacity_;
  size_t                                                  size_;
  temporary_file_t                                        file_;
  std::shared_ptr<char>                                   base_;

  std::shared_ptr<background_executor>                    bio_;
  std::atomic<size_t>                                     traverse_i_;
  std::vector<object_buffer_detail::traverse_unit<char>>  traverse_range_;
  std::atomic<size_t>                                     traverse_prefetch_i_;
  std::vector<size_t>                                     traverse_anchor_;
  bool                                                    traverse_direction_;
};

// ******************************************************************************* //
// implementations

template <typename T, typename Enable>
object_file_buffer_t<T, Enable>::object_file_buffer_t(size_t capacity, std::string cache_dir) :
    size_(0), file_(cache_dir), bio_(new background_executor()), traverse_direction_(true) {
  if (0 == capacity) {
    boost::filesystem::create_directories(cache_dir);
    capacity = std::min(512UL * GBYTES, boost::filesystem::space(cache_dir).capacity);
  }
  capacity_ = capacity;
  CHECK(capacity_);

  CHECK(-1 != ftruncate64(file_.fd(), capacity_))
  << boost::format("WARNING: ftruncate64 failed, err code: %d, err msg: %s") % errno % strerror(errno);
  CHECK(-1 != fsync(file_.fd()))
  << boost::format("WARNING: fsync failed, err code: %d, err msg: %s") % errno % strerror(errno);

  base_.reset((char*)mmap(0, capacity_, PROT_WRITE, MAP_SHARED | MAP_NORESERVE, file_.fd(), 0), [capacity] (char* p) {
    CHECK(-1 != munmap(p, capacity))
    << boost::format("WARNING: munmap failed, err code: %d, err msg: %s") % errno % strerror(errno);
  });
  CHECK(MAP_FAILED != base_.get())
  << boost::format("WARNING: mmap failed, err code: %d, err msg: %s.") % errno % strerror(errno);
  CHECK(-1 != madvise(base_.get(), capacity_, MADV_SEQUENTIAL))
  << boost::format("WARNING: madvise failed, err code: %d, err msg: %s") % errno % strerror(errno);
  traverse_anchor_.resize(capacity_ / mmap_unit_capacity + 1, 0);
}

template <typename T, typename Enable>
object_file_buffer_t<T, Enable>::object_file_buffer_t(object_buffer_opt_t opt) :
    object_file_buffer_t(opt.capacity_, opt.path_) { }

template <typename T, typename Enable>
object_file_buffer_t<T, Enable>& object_file_buffer_t<T, Enable>::operator=(object_file_buffer_t&& x) noexcept {
  capacity_           = x.capacity_;
  size_               = x.size_;
  file_               = std::move(x.file_);
  base_               = std::move(x.base_);
  traverse_direction_ = x.traverse_direction_;

  traverse_i_.store(0);
  traverse_range_.clear();
  traverse_prefetch_i_.store(0);
  traverse_anchor_ = std::move(x.traverse_anchor_);

  x.size_     = 0;
  x.capacity_ = 0;
  x.traverse_range_.clear();

  return *this;
}

template <typename T, typename Enable>
object_file_buffer_t<T, Enable>::object_file_buffer_t(object_file_buffer_t&& other) noexcept {
  this->operator=(std::forward<object_file_buffer_t>(other));
}

template <typename T, typename Enable>
size_t object_file_buffer_t<T, Enable>::push_back(const T* pitems, size_t n) {
  using empty_oarchive_t = yas::binary_oarchive<empty_ostream_t, yas::binary | yas::ehost | yas::no_header>;
  using mem_simple_oarchive_t = yas::binary_oarchive<mem_simple_ostream_t, yas::binary | yas::ehost | yas::no_header>;

  size_t ser_size;
  {
    empty_ostream_t empty_ostream;
    empty_oarchive_t empty_oarchive(empty_ostream);
    for (size_t i = 0; i < n; ++i) {
      empty_oarchive & pitems[i];
    }
    ser_size = empty_ostream.size();
  }

  size_t offset_begin = __sync_fetch_and_add(&size_, ser_size);
  size_t offset_end = offset_begin + ser_size;
  size_t unit_index_end = boost::alignment::align_up(offset_end, mmap_unit_capacity) / mmap_unit_capacity;
  CHECK(offset_end <= capacity_)
  << "object buffer overflow, offset_begin: " << offset_begin << ", ser_size: " << ser_size << ", capacity_: " << capacity_;

  {
    mem_simple_ostream_t mem_simple_ostream(base_.get() + offset_begin, base_.get() + offset_end);
    mem_simple_oarchive_t mem_simple_oarchive(mem_simple_ostream);
    for (size_t i = 0; i < n; ++i) {
      mem_simple_oarchive & pitems[i];
    }
    CHECK(mem_simple_ostream.size() == ser_size);
  }

  write_max(&traverse_anchor_[unit_index_end - 1], offset_end);

  return offset_begin;
}

template <typename T, typename Enable>
void object_file_buffer_t<T, Enable>::reset_traversal(const traverse_opts_t&) {
  {
    traverse_direction_ = !traverse_direction_;
    mprotect(base_.get(), capacity_, PROT_NONE);
    auto base = base_;
    auto size = size_;
    auto capacity = capacity_;
    bio_->submit([base, size, capacity] {
      CHECK(-1 != msync(base.get(), size, MS_SYNC))
      << boost::format("WARNING: msync failed, err code: %d, err msg: %s") % errno % strerror(errno);
      CHECK(-1 != madvise(base.get(), capacity, MADV_DONTNEED))
      << boost::format("WARNING: madvise failed, err code: %d, err msg: %s") % errno % strerror(errno);
    });
  }

  traverse_i_ = 0;
  traverse_prefetch_i_ = 0;
  size_t buckets = size_ / mmap_unit_capacity + (size_ % mmap_unit_capacity ? 1 : 0);

  traverse_range_.clear();
  traverse_range_.reserve(buckets);
  size_t size = 0;
  for (size_t anchor : traverse_anchor_) {
    if (anchor) {
      traverse_range_.emplace_back(file_.fd(), size, anchor);
      size = anchor;
    }
  }
  if (!traverse_direction_) {
    std::reverse(traverse_range_.begin(), traverse_range_.end());
  }
  CHECK(size == size_);
}

template <typename T, typename Enable>
template <typename Traversal>
bool object_file_buffer_t<T, Enable>::next_chunk(Traversal&& traversal, size_t* chunk_size) {
  size_t range_i = traverse_i_.fetch_add(1);
  if (range_i >= traverse_range_.size()) return false;
  *chunk_size = 1;

  bio_->submit([this] {
    size_t traverse_prefetch_i = traverse_prefetch_i_.fetch_add(1);
    if (traverse_prefetch_i < traverse_range_.size()) {
      if (!traverse_range_[traverse_prefetch_i].prefetch(false)) {
        // prefecth failed. extend prefetch windows.
        traverse_prefetch_i = traverse_prefetch_i_.fetch_add(1);
        if (traverse_prefetch_i < traverse_range_.size()) {
          traverse_range_[traverse_prefetch_i].prefetch(false);
        }
      }
    }
  });

  object_buffer_detail::traverse_unit<char>& traverse = traverse_range_[range_i];
  traverse.prefetch(true);

  CHECK(traverse.objs_begin_);

  using mem_iarchive_t = yas::binary_iarchive<mem_istream_t, yas::binary | yas::ehost | yas::no_header>;
  mem_istream_t mem_istream(traverse.objs_begin_, traverse.objs_end_ - traverse.objs_begin_);
  mem_iarchive_t mem_iarchive(mem_istream);

  while(!mem_istream.empty()) {
    T item;
    mem_iarchive & item;
    traversal(std::numeric_limits<size_t>::max(), &item);
  }

  traverse.evict();
  return true;
}

// ******************************************************************************* //

// fixed-size, object file buffer with thread-safe traversal
template <typename T>
class object_file_buffer_t<T, typename std::enable_if<std::is_trivial<T>::value>::type> {
  constexpr static size_t mmap_unit_capacity = 4 * MBYTES;
public:
  size_t size(void) { return size_; }
  size_t capacity(void) { return capacity_; }

  // thread-safe push_back
  size_t push_back(const T& item) { return push_back(&item, 1); }

  size_t push_back(const T* pitems, size_t n);

  // traversal related

  // not thread-safe
  void reset_traversal(const traverse_opts_t& = traverse_opts_t());

  /*
   * process a chunk of objects, thread-safe
   *
   * \param traversal    process callback functor, shoud implement
   *                     <tt> void operator()(size_t idx, T* object) <\tt>
   * \param chunk_size   input, traverse at most chunk_size objects,
   *                     output, real objects processed
   *
   * \return
   *    true  -- traverse at lease one object
   *    false -- no more object to traverse
   * */
  template <typename Traversal>
  bool next_chunk(Traversal&& traversal, size_t* chunk_size);

  // ********************* end of required types & methods ************************** //
  explicit object_file_buffer_t(
      size_t capacity = 0,
      std::string cache_dir = ".cache/");

  explicit object_file_buffer_t(object_buffer_opt_t opt);

  object_file_buffer_t(object_file_buffer_t&& other) noexcept;
  object_file_buffer_t& operator=(object_file_buffer_t&& x) noexcept;

  object_file_buffer_t(const object_file_buffer_t&) = delete;
  object_file_buffer_t& operator=(const object_file_buffer_t&) = delete;

  constexpr static bool is_trivial() { return std::is_trivial<T>::value; }
protected:
  size_t                                                capacity_;
  size_t                                                size_;
  temporary_file_t                                      file_;
  std::shared_ptr<T>                                    base_;

  std::shared_ptr<background_executor>                  bio_;
  std::atomic<size_t>                                   traverse_i_;
  std::vector<object_buffer_detail::traverse_unit<T>>   traverse_range_;
  std::atomic<size_t>                                   traverse_prefetch_i_;
  bool                                                  traverse_direction_;
};

// ******************************************************************************* //
// implementations

template <typename T>
object_file_buffer_t<T, typename std::enable_if<std::is_trivial<T>::value>::type>::
object_file_buffer_t(size_t capacity, std::string cache_dir) :
    size_(0), file_(cache_dir), bio_(new background_executor()), traverse_direction_(true) {
  if (0 == capacity) {
    boost::filesystem::create_directories(cache_dir);
    capacity = std::min(512UL * GBYTES, boost::filesystem::space(cache_dir).capacity) / sizeof(T);
  }

  capacity_ = capacity;
  CHECK(capacity_);

  CHECK(-1 != ftruncate64(file_.fd(), sizeof(T) * capacity_))
  << boost::format("WARNING: ftruncate64 failed, err code: %d, err msg: %s") % errno % strerror(errno);
  CHECK(-1 != fsync(file_.fd()))
  << boost::format("WARNING: fsync failed, err code: %d, err msg: %s") % errno % strerror(errno);

  base_.reset((T*)mmap(0, sizeof(T) * capacity_, PROT_WRITE, MAP_SHARED | MAP_NORESERVE, file_.fd(), 0), [capacity] (T* p) {
    CHECK(-1 != munmap(p, sizeof(T) * capacity))
    << boost::format("WARNING: munmap failed, err code: %d, err msg: %s") % errno % strerror(errno);
  });
  CHECK(MAP_FAILED != base_.get())
  << boost::format("WARNING: mmap failed, err code: %d, err msg: %s.") % errno % strerror(errno);
  CHECK(-1 != madvise(base_.get(), sizeof(T) * capacity_, MADV_SEQUENTIAL))
  << boost::format("WARNING: madvise failed, err code: %d, err msg: %s") % errno % strerror(errno);
}

template <typename T>
object_file_buffer_t<T, typename std::enable_if<std::is_trivial<T>::value>::type>::
object_file_buffer_t(object_buffer_opt_t opt) :
    object_file_buffer_t(opt.capacity_, opt.path_) { }

template <typename T>
object_file_buffer_t<T, typename std::enable_if<std::is_trivial<T>::value>::type>&
object_file_buffer_t<T, typename std::enable_if<std::is_trivial<T>::value>::type>::
operator=(object_file_buffer_t&& x) noexcept {
  capacity_           = x.capacity_;
  size_               = x.size_;
  file_               = std::move(x.file_);
  base_               = std::move(x.base_);
  bio_                = std::move(x.bio_);
  traverse_direction_ = x.traverse_direction_;

  traverse_i_.store(0);
  traverse_range_.clear();
  traverse_prefetch_i_.store(0);

  x.size_     = 0;
  x.capacity_ = 0;
  x.traverse_range_.clear();

  return *this;
}

template <typename T>
object_file_buffer_t<T, typename std::enable_if<std::is_trivial<T>::value>::type>::
object_file_buffer_t(object_file_buffer_t&& other) noexcept {
  this->operator=(std::forward<object_file_buffer_t>(other));
}

template <typename T>
size_t
object_file_buffer_t<T, typename std::enable_if<std::is_trivial<T>::value>::type>::
push_back(const T* pitems, size_t n) {
  size_t offset_begin = __sync_fetch_and_add(&size_, n);
  CHECK(offset_begin + n <= capacity_)
  << "object buffer overflow, offset_begin: " << offset_begin << ", n: " << n << ", capacity_: " << capacity_;

  memcpy(base_.get() + offset_begin, pitems, sizeof(T) * n);
  return offset_begin;
}

template <typename T>
void
object_file_buffer_t<T, typename std::enable_if<std::is_trivial<T>::value>::type>::
reset_traversal(const traverse_opts_t&) {
  {
    traverse_direction_ = !traverse_direction_;
    mprotect(base_.get(), capacity_ * sizeof(T), PROT_NONE);
    auto base = base_;
    auto size = size_;
    auto capacity = capacity_;
    bio_->submit([base, size, capacity] {
      CHECK(-1 != msync(base.get(), size * sizeof(T), MS_SYNC))
      << boost::format("WARNING: msync failed, err code: %d, err msg: %s") % errno % strerror(errno);
      CHECK(-1 != madvise(base.get(), capacity * sizeof(T), MADV_DONTNEED))
      << boost::format("WARNING: madvise failed, err code: %d, err msg: %s") % errno % strerror(errno);
    });
  }

  traverse_i_ = 0;
  traverse_prefetch_i_ = 0;
  size_t buckets = size_ / mmap_unit_capacity + (size_ % mmap_unit_capacity ? 1 : 0);

  traverse_range_.clear();
  traverse_range_.reserve(buckets);
  for (size_t i = 0; i < buckets; ++i) {
    traverse_range_.emplace_back(file_.fd(), mmap_unit_capacity * i, std::min(mmap_unit_capacity * (i + 1), size_));
  }
  if (!traverse_direction_) {
    std::reverse(traverse_range_.begin(), traverse_range_.end());
  }
}

template <typename T>
template <typename Traversal>
bool
object_file_buffer_t<T, typename std::enable_if<std::is_trivial<T>::value>::type>::
next_chunk(Traversal&& traversal, size_t* chunk_size) {
  size_t range_i = traverse_i_.fetch_add(1);
  if (range_i >= traverse_range_.size()) return false;
  *chunk_size = 1;

  bio_->submit([this] {
    size_t traverse_prefetch_i = traverse_prefetch_i_.fetch_add(1);
    if (traverse_prefetch_i < traverse_range_.size()) {
      if (!traverse_range_[traverse_prefetch_i].prefetch(false)) {
        // prefecth failed. extend prefetch windows.
        traverse_prefetch_i = traverse_prefetch_i_.fetch_add(1);
        if (traverse_prefetch_i < traverse_range_.size()) {
          traverse_range_[traverse_prefetch_i].prefetch(false);
        }
      }
    }
  });

  object_buffer_detail::traverse_unit<T>& traverse = traverse_range_[range_i];
  traverse.prefetch(true);

  CHECK(traverse.objs_begin_);

  for (T* p = traverse.objs_begin_; p < traverse.objs_end_; ++p) {
    traversal(std::numeric_limits<size_t>::max(), p);
  }

  traverse.evict();
  return true;
}

// ******************************************************************************* //

namespace object_buffer_detail {

template <typename T>
class object_dfs_buffer_base_t {
public:
  void reset_traversal() {
    if (dfs_) {
      dfs_->foreach([this] (const std::string& filename, boost::iostreams::filtering_ostream& /* os */) {
        files_.push_back(filename);
      });
      dfs_.reset();
    }
    files_traverse_ = files_;
  }

  bool fetch_file(std::string& filename) {
    std::lock_guard<std::mutex> lock(*mutex_);
    if (files_traverse_.empty()) {
      return false;
    }
    filename = std::move(files_traverse_.front());
    files_traverse_.pop_front();
    return true;
  }

  constexpr static bool is_trivial() { return std::is_trivial<T>::value; }

  explicit object_dfs_buffer_base_t(std::string prefix, std::string cache_dir = ".cache") :
      prefix_(std::move(prefix)), cache_dir_(std::move(cache_dir)) {
    mutex_.reset(new std::mutex);
    dfs_.reset(new thread_local_fs_output(cache_dir_, prefix_, false));
  }

  explicit object_dfs_buffer_base_t(object_buffer_opt_t opt) {
    if (opt.path_.empty()) {
      opt.path_ = ".cache";
    }
    prefix_ = opt.prefix_;
    cache_dir_ = opt.path_;
    mutex_.reset(new std::mutex);
    dfs_.reset(new thread_local_fs_output(cache_dir_, prefix_, false));
  }

  ~object_dfs_buffer_base_t() {
    dfs_.reset();

    if (boost::istarts_with(cache_dir_, "hdfs://")) {
      for (auto& file : files_) {
        CHECK(0 == hdfs_t::get_hdfs(file).remove(file, 1));
      }
    } else {
      for (auto& file : files_) {
        CHECK(boost::filesystem::remove(file));
      }
    }
  }

  object_dfs_buffer_base_t(const object_dfs_buffer_base_t&) = delete;
  object_dfs_buffer_base_t& operator=(const object_dfs_buffer_base_t&) = delete;
  object_dfs_buffer_base_t(object_dfs_buffer_base_t&& other) = default;
  object_dfs_buffer_base_t& operator=(object_dfs_buffer_base_t&& x) = default;

protected:
  std::string                             prefix_;
  std::string                             cache_dir_;
  std::list<std::string>                  files_;
  std::list<std::string>                  files_traverse_;
  std::unique_ptr<std::mutex>             mutex_;
  std::unique_ptr<thread_local_fs_output> dfs_;
};

}

// unlimited-size, object file buffer with thread-safe traversal
template <typename T, typename Enable = void>
class object_dfs_buffer_t : public object_buffer_detail::object_dfs_buffer_base_t<T> {
public:
  size_t push_back(const T& item) { return push_back(&item, 1); }

  size_t push_back(const T* pitems, size_t n) {
    boost::iostreams::filtering_ostream& os = this->dfs_->local();
    filtering_ostream_t ostream(os);
    yas::binary_oarchive<filtering_ostream_t, yas::binary | yas::ehost | yas::no_header> oar(ostream);
    for (size_t i = 0; i < n; ++i) {
      oar & pitems[i];
    }
    return std::numeric_limits<size_t >::max();
  }

  template <typename Traversal>
  bool next_chunk(Traversal&& traversal, size_t* chunk_size) {
    std::string filename;
    if (!this->fetch_file(filename)) return false;

    std::unique_ptr<hdfs_t::fstream> hdfs_fs;
    boost::iostreams::filtering_istream is;

    if (boost::istarts_with(filename, "hdfs://")) {
      hdfs_fs.reset(new hdfs_t::fstream(hdfs_t::get_hdfs(filename), filename));
      is.push(*hdfs_fs);
    } else {
      is.push(boost::iostreams::file_source(filename));
    }

    filtering_istream_t istream(is);
    yas::binary_iarchive<filtering_istream_t, yas::binary | yas::ehost | yas::no_header> iar(istream);

    while (!iar.empty()) {
      T item;
      iar & item;
      if (__glibc_unlikely(!is.good())) break;
      traversal(std::numeric_limits<size_t>::max(), &item);
    }

    *chunk_size = 1;
    return true;
  }

  explicit object_dfs_buffer_t(std::string prefix, std::string cache_dir = ".cache") :
      object_buffer_detail::object_dfs_buffer_base_t<T>(std::move(prefix), std::move(cache_dir)) {}
  explicit object_dfs_buffer_t(object_buffer_opt_t opt) :
      object_buffer_detail::object_dfs_buffer_base_t<T>(std::move(opt)) { }
};

template <typename T>
class object_dfs_buffer_t<T, typename std::enable_if<std::is_trivial<T>::value>::type> :
    public object_buffer_detail::object_dfs_buffer_base_t<T> {
public:
  size_t push_back(const T& item) { return push_back(&item, 1); }

  size_t push_back(const T* pitems, size_t n) {
    boost::iostreams::filtering_ostream& os = this->dfs_->local();
    os.write((char*)pitems, sizeof(T) * n);
    return std::numeric_limits<size_t >::max();
  }

  template <typename Traversal>
  bool next_chunk(Traversal&& traversal, size_t* chunk_size) {
    std::string filename;
    if (!this->fetch_file(filename)) return false;

    std::unique_ptr<hdfs_t::fstream> hdfs_fs;
    boost::iostreams::filtering_istream is;

    if (boost::istarts_with(filename, "hdfs://")) {
      hdfs_fs.reset(new hdfs_t::fstream(hdfs_t::get_hdfs(filename), filename));
      is.push(*hdfs_fs);
    } else {
      is.push(boost::iostreams::file_source(filename));
    }

    while (!is.eof()) {
      T item;
      is.read((char*)&item, sizeof(T));
      if (__glibc_unlikely(!is.good())) break;
      traversal(std::numeric_limits<size_t>::max(), &item);
    }

    *chunk_size = 1;
    return true;
  }

  explicit object_dfs_buffer_t(std::string prefix, std::string cache_dir = ".cache") :
      object_buffer_detail::object_dfs_buffer_base_t<T>(std::move(prefix), std::move(cache_dir)) {}
  explicit object_dfs_buffer_t(object_buffer_opt_t opt) :
      object_buffer_detail::object_dfs_buffer_base_t<T>(std::move(opt)) { }
};

}  // namespace plato

