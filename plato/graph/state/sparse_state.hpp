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

#ifndef __PLATO_GRAPH_SPARSE_STATE_HPP__
#define __PLATO_GRAPH_SPARSE_STATE_HPP__

#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>
#include <functional>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "libcuckoo/cuckoohash_map.hh"

#include "plato/graph/base.hpp"
#include "plato/graph/state/detail.hpp"
#include "plato/util/bitmap.hpp"
#include "plato/util/hash.hpp"

namespace plato {

template <typename T, typename PART_IMPL, typename HASH = cuckoo_vid_hash, typename KEY_EQUAL = std::equal_to<vid_t>,
  typename ALLOC = std::allocator<std::pair<const vid_t, T>>, typename BITMAP = bitmap_t<>>
class sparse_state_t {
public:

  using hash_t      = HASH;
  using key_equal_t = KEY_EQUAL;

  // ******************************************************************************* //
  // required types & methods

  using value_t        = T;
  using partition_t    = PART_IMPL;
  using allocator_type = ALLOC;
  using bitmap_spec_t  = BITMAP;

  size_t count(vid_t v_i) const;

  /*
   * returns a reference to the element belong to v_i
   * if v_i's value not exists, insert a new value and return it
   *
   **/
  T&       operator[] (vid_t v_i);
  const T& operator[] (vid_t v_i) const;

  // get partitioner
  std::shared_ptr<partition_t> partitioner(void) { return partitioner_; }

  // start travel from all/subset of the vertices, not thread-safe
  void reset_traversal(std::shared_ptr<bitmap_spec_t> active = nullptr);

  /*
   * process a chunk of vertices, thread-safe
   *
   * \tparam F  type of the functor. It should implement the method
   *            <tt>bool operator()(vid_t, T*)</tt>
   *
   * \param traversal    callback function process on each vertex's state
   * \param chunk_size   process at most chunk_size vertices
   *
   * \return
   *    true  -- process at lease one vertex
   *    false -- no more vertex to process
   **/
  template <typename F>
  bool next_chunk(F&& traversal, size_t* chunk_size);

  /*
   * process active vertices in parallel
   *
   * \param process     user define callback for each eligible vertex
   *                    R(vid_t v_i, value_t* value)
   * \param pactives    bitmap used for filter subset of the vertex
   * \param chunk_size  at most process 'chunk_size' chunk at a batch
   *
   * \return
   *        sum of 'process' return
   **/
  template <typename R, typename PROCESS>
  R foreach(PROCESS&& process, bitmap_spec_t* pactives = nullptr, size_t chunk_size = PAGESIZE);

  // ******************************************************************************* //

  /* Constructor
   *
   * \param n      the number of elements reserve space for initially
   * \param part   the partitioner of state
   * \param hfunc  hash function instance to use
   * \param equal  equality function instance to use
   * \param alloc  allocator instance to use
   **/
  sparse_state_t(size_t n, std::shared_ptr<partition_t> part, const HASH& hfunc = HASH(),
      const KEY_EQUAL& equal = KEY_EQUAL(), const ALLOC& alloc = ALLOC());
  sparse_state_t(sparse_state_t&&);

  sparse_state_t(const sparse_state_t&) = delete;
  sparse_state_t& operator=(const sparse_state_t&) = delete;

  // return lock_table_'s size
  size_t size(void);

  /*
   * Searches for 'v_i' in the table. If the key is found, then 'func' is
   * called on the existing value, and nothing happens to the passed-in 'v_i' and
   * values. The functor can mutate the value. If the key is not
   * found and must be inserted, the pair will be constructed by forwarding the
   * given key and values.
   *
   * \tparam F     type of the functor. It should implement the method
   *               <tt>bool operator()(T&)</tt>
   * \tparam Args  list of types for the value constructor arguments
   *
   * \param v_i    the key to insert into the table
   * \param func   the functor to invoke if the element is found
   * \param val    a list of constructor arguments with which to create the value
   *
   * \return true if a new key was inserted, false if the key was already in
   *  the table
   **/
  template <typename F, typename... Args>
  bool upsert(vid_t v_i, F&& func, Args&&... val);

  /**
   * Inserts the key-value pair into the table. Equivalent to calling @ref
   * upsert with a functor that does nothing.
   */
  template <typename... Args>
  bool insert(vid_t v_i, Args&&... val);

  /*
   * \tparam F  type of the functor. It should implement the method
   *            <tt>void operator()(T&)</tt>
   **/
  template <typename F>
  bool update(vid_t v_i, F&& func);

  // removes all elements in the table, calling their destructors.
  void clear(void);

  /*
   * lock the table, enable operator[] and traversal operations
   * sparse-state was initialized in unlock state
   **/
  void lock(void);

  /*
   * release the lock, disable operator[] and traversal operations
   * sparse-state was initialized in unlock state
   **/
  void unlock(void);

  /*
   * Searches for 'v_i' in the table. and invokes @p fn on the value. @p fn is
   * not allowed to modify the contents of the value if found.
   *
   * \tparam F     type of the functor. It should implement the method
   *               <tt>auto operator()(T&)</tt>
   *
   * \param v_i    the key to search for
   * \param fn     the functor to invoke if the element is found
   *
   * \return true if the key was found and functor invoked, false otherwise
   **/
  template <typename F>
  bool find_fn(vid_t v_i, F fn) const;

protected:
  using cuckoomap_t    = cuckoohash_map<vid_t, value_t, hash_t, key_equal_t, allocator_type>;
  using locked_table_t = typename cuckoomap_t::locked_table;
  using iterator_t     = typename locked_table_t::iterator;

  cuckoomap_t data_;
  std::shared_ptr<partition_t> partitioner_;
  std::unique_ptr<locked_table_t> lock_table_;

  // traversal related
  bool is_dirty_;
  static const size_t basic_chunk = 64;
  std::atomic<size_t> traverse_i_;
  std::vector<std::pair<iterator_t, iterator_t>> traverse_range_;
  std::shared_ptr<bitmap_spec_t> traverse_active_;
};

// ************************************************************************************ //
// implementations

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::sparse_state_t(size_t n, std::shared_ptr<partition_t> part,
    const HASH& hfunc, const KEY_EQUAL& equal, const ALLOC& alloc)
  : data_(n, hfunc, equal, alloc), partitioner_(part), lock_table_(nullptr), is_dirty_(true) { }

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::sparse_state_t(sparse_state_t&& other)
  : data_(std::move(other.data_)), partitioner_(other.partitioner_), lock_table_(std::move(other.lock_table_)),
    is_dirty_(other.is_dirty_), traverse_i_(0), traverse_range_(std::move(other.traverse_range_)),
    traverse_active_(std::move(other.traverse_active_)) { }

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
template <typename F, typename... Args>
bool sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::upsert(vid_t v_i, F&& func, Args&&... val) {
  CHECK(nullptr == lock_table_) << "can not upsert when state is lock";
  is_dirty_ = true;
  return data_.upsert(v_i, std::forward<F>(func), std::forward<Args>(val)...);
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
template <typename... Args>
bool sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::insert(vid_t v_i, Args&&... val) {
  return upsert(v_i, [](value_t&) {}, std::forward<Args>(val)...);
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
template <typename F>
bool sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::update(vid_t v_i, F&& func) {
  CHECK(nullptr == lock_table_) << "can not update when state is lock";
  is_dirty_ = true;
  return data_.update_fn(v_i, std::forward<F>(func));
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
void sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::clear(void) {
  CHECK(nullptr == lock_table_) << "can not clear when state is lock";
  is_dirty_ = true;
  data_.clear();
  traverse_i_.store(0, std::memory_order_relaxed);
  traverse_range_.clear();
  traverse_active_.reset();
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
size_t sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::size(void) {
  CHECK(lock_table_) << "can not query table size when state is not locked";
  return lock_table_->size();
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
void sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::lock(void) {
  if (lock_table_) { return ; }
  lock_table_.reset(new locked_table_t(std::move(data_.lock_table())));
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
void sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::unlock(void) {
  lock_table_.reset(nullptr);
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
template <typename F>
bool sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::find_fn(vid_t v_i, F fn) const {
  CHECK(lock_table_) << "can not call find_fn when table is not locked";
  auto iter = lock_table_->find(v_i);
  if (iter != lock_table_->end()) {
    fn(iter->second);
    return true;
  }
  return false;
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
T& sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::operator[] (vid_t v_i) {
  CHECK(lock_table_) << "can not call operator[] when table is not locked";
  return lock_table_->at(v_i);
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
const T& sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::operator[] (vid_t v_i) const {
  CHECK(lock_table_) << "can not call operator[] when table is not locked";
  return lock_table_->at(v_i);
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
size_t sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::count(vid_t v_i) const {
  return lock_table_->count(v_i);
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
void sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::reset_traversal(std::shared_ptr<bitmap_spec_t> active) {
  traverse_active_ = active;
  traverse_i_.store(0, std::memory_order_relaxed);

  if (false == is_dirty_) { return ; }
  if (nullptr == lock_table_) { lock(); }

  traverse_range_.clear();
  size_t buckets = std::min((size_t)(MBYTES), (size_t)lock_table_->size() / basic_chunk + 1);

  {
    traverse_range_.resize(buckets);

    iterator_t it    = lock_table_->begin();
    size_t remaining = lock_table_->size();

    for (size_t i = 0; i < buckets; ++i) {
      if (i == (buckets - 1)) {
        traverse_range_[i].first  = it;
        traverse_range_[i].second = lock_table_->end();
      } else {
        size_t amount = remaining / (buckets - i) / basic_chunk * basic_chunk;

        traverse_range_[i].first = it;
        for (size_t j = 0; j < amount && lock_table_->end() != it; ++j, ++it) { }
        traverse_range_[i].second = it;

        remaining -= amount;
      }
    }
  }
  is_dirty_ = false;
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
template <typename F>
bool sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::next_chunk(F&& traversal, size_t* chunk_size) {
  size_t range_start = traverse_i_.fetch_add(*chunk_size, std::memory_order_relaxed);
  if (range_start >= traverse_range_.size()) { return false; }
  if (range_start + *chunk_size > traverse_range_.size()) {
    *chunk_size = traverse_range_.size() - range_start;
  }
  size_t range_end = range_start + *chunk_size;

  for (size_t range_i = range_start; range_end != range_i; ++range_i) {
    iterator_t __begin = traverse_range_[range_i].first;
    iterator_t __end   = traverse_range_[range_i].second;

    for (iterator_t it = __begin; __end != it; ++it) {
      if (traverse_active_ && !traverse_active_->get_bit(it->first)) { continue; }
      traversal(it->first, &it->second);
    }
  }

  return true;
}

template <typename T, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
template <typename R, typename PROCESS>
R sparse_state_t<T, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::foreach(PROCESS&& process, bitmap_spec_t* pactives,
    size_t chunk_size) {
  return __foreach<R>(this, std::forward<PROCESS>(process), pactives, chunk_size);
}

// ************************************************************************************ //

}

#endif

