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

#ifndef __PLATO_GRAPH_DENSE_STATE_HPP__
#define __PLATO_GRAPH_DENSE_STATE_HPP__

#include <cstdint>
#include <memory>
#include <atomic>
#include <random>
#include <type_traits>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "plato/graph/base.hpp"
#include "plato/graph/state/detail.hpp"
#include "plato/graph/partition/sequence.hpp"
#include "plato/util/bitmap.hpp"
#include "plato/util/mmap_alloc.hpp"

namespace plato {

template <typename T, typename PART_IMPL, typename ALLOC = mmap_allocator_t<T>, typename BITMAP = bitmap_t<>>
class dense_state_t {
protected:
  using traits_ = typename std::allocator_traits<ALLOC>::template rebind_traits<T>;

public:

  // static_assert(std::is_trivial<T>::value && std::is_standard_layout<T>::value,
  //     "dense_state_t only support pod-type");


  // ******************************************************************************* //
  // required types & methods

  using value_t        = T;
  using partition_t    = PART_IMPL;
  using allocator_type = typename traits_::allocator_type;
  using bitmap_spec_t  = BITMAP;

  /*
   * call 'func' on each vertices belong to this partition
   *
   * \param func    user provide vertex process logic
   * \param active  active vertices set
   *
   * \return
   *    sum of every func's return value
   **/
  // template <typename R>
  // R foreach(std::function<R(vid_t, T*)> func, bitmap_t active);

  // fill all vertices value to T belong to this partition
  void fill(const T& value);

  /*
   * returns a reference to the element at position n in the state container
   **/
  T&       operator[] (size_t n);
  const T& operator[] (size_t n) const;

  // get partitioner
  std::shared_ptr<partition_t> partitioner(void) { return partitioner_; }

  // traverse related

  // return true -- continue travel, false -- stop.
  using traversal_t = std::function<bool(vid_t, T*)>;

  // start travel from all/subset of the vertices
  void reset_traversal(std::shared_ptr<bitmap_spec_t> pactive = nullptr,
      const traverse_opts_t& opts = traverse_opts_t());

  /*
   * process a chunk of vertices
   *
   * \param traversal   callback function process on each vertex's state
   * \param chunk_size  process at most chunk_size vertices
   *
   * \return 
   *    true  -- process at lease one vertex
   *    false -- no more vertex to process
   **/
  bool next_chunk(traversal_t traversal, size_t* chunk_size);

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

  /*
   * constructor
   *
   * \param max_v_i       maximum vertex's id
   * \param partitioner
   * \param alloc         allocator for internal storage
   **/
  dense_state_t(vid_t max_v_i, std::shared_ptr<partition_t> partitioner,
      const allocator_type& alloc = ALLOC());

  dense_state_t(dense_state_t&& other);
  dense_state_t& operator=(dense_state_t&& other);

  dense_state_t(const dense_state_t&) = delete;
  dense_state_t& operator=(const dense_state_t&) = delete;

  ~dense_state_t(void);

  // std::shared_ptr<T> data(void) { return data_; }
  allocator_type& allocator(void) { return allocator_; }

protected:

  vid_t max_v_i_;
  allocator_type allocator_;

  value_t*                       data_;
  std::shared_ptr<partition_t>   partitioner_;

  vid_t                                traverse_start_;
  vid_t                                traverse_end_;
  std::atomic<vid_t>                   traverse_i_;
  std::shared_ptr<bitmap_spec_t>       traverse_active_;
  std::vector<std::pair<vid_t, vid_t>> traverse_range_;
  traverse_opts_t                      traverse_opts_;

  const cluster_info_t&          cluster_info_;

  // SFINAE only works for deduced template arguments, it's tricky here
  template <typename PART>
  typename std::enable_if<is_seq_part<PART>(), vid_t>::type part_start(std::shared_ptr<PART> part) {
    return part->offset_[cluster_info_.partition_id_];
  }
  template <typename PART>
  typename std::enable_if<is_seq_part<PART>(), vid_t>::type part_end(std::shared_ptr<PART> part) {
    return part->offset_[cluster_info_.partition_id_ + 1];
  }

  template <typename PART>
  typename std::enable_if<!is_seq_part<PART>(), vid_t>::type part_start(std::shared_ptr<PART>) { return 0; }
  template <typename PART>
  typename std::enable_if<!is_seq_part<PART>(), vid_t>::type part_end(std::shared_ptr<PART>)   { return max_v_i_; }
};

// ************************************************************************************ //
// implementations

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::dense_state_t(vid_t max_v_i, std::shared_ptr<partition_t> partitioner,
    const allocator_type& alloc)
  : max_v_i_(max_v_i + 1), allocator_(alloc), partitioner_(partitioner),
    traverse_start_(0), traverse_end_(0), traverse_i_(0),
    cluster_info_(cluster_info_t::get_instance())
{
  data_ = allocator_.allocate(max_v_i_);
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::dense_state_t(dense_state_t&& ot)
  : max_v_i_(ot.max_v_i_), allocator_(ot.allocator_), partitioner_(std::move(ot.partitioner_)),
    traverse_start_(ot.traverse_start_), traverse_end_(ot.traverse_end_), traverse_i_(ot.traverse_i_.load()),
    traverse_active_(std::move(ot.traverse_active_)), cluster_info_(ot.cluster_info_)
{
  data_    = ot.data_;
  ot.data_ = nullptr;
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
dense_state_t<T, PART_IMPL, ALLOC, BITMAP>&
dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::operator=(dense_state_t&& ot) {
  if (nullptr != data_) {
    allocator_.deallocate(data_, max_v_i_);
  }

  max_v_i_        = ot.max_v_i_;
  allocator_      = ot.allocator_;
  partitioner_    = std::move(ot.partitioner_);
  traverse_start_ = ot.traverse_start_;
  traverse_end_   = ot.traverse_end_;
  traverse_i_.store(ot.traverse_i_);
  traverse_active_ = std::move(ot.traverse_active_);

  data_    = ot.data_;
  ot.data_ = nullptr;

  // cluster_info_ ??
  return *this;
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::~dense_state_t(void) {
  if (nullptr != data_) {
    allocator_.deallocate(data_, max_v_i_);
  }
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
inline T& dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::operator[] (size_t n) {
  // CHECK(n < max_v_i_);
  return data_[n];
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
inline const T& dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::operator[] (size_t n) const {
  // CHECK(n < max_v_i_);
  return data_[n];
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
void dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::fill(const T& value) {
  auto traversal = [&](vid_t v_i, T* pval) {
    *pval = value;
    return true;
  };

  reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 64;
    while (next_chunk(traversal, &chunk_size)) { }
  }
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
void dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::reset_traversal(std::shared_ptr<bitmap_spec_t> pactive,
    const traverse_opts_t& opts) {
  if (false == is_seq_part<partition_t>()) {
    LOG(WARNING) << "scan all vertices to perform foreach op. sequence partition/sparse state is a better option";
  }

  traverse_start_  = part_start(partitioner_);
  traverse_end_    = part_end(partitioner_);
  traverse_active_ = pactive;

  traverse_i_.store(traverse_start_, std::memory_order_relaxed);

  // prebuild chunk blocks for random traverse
  if (is_seq_part<partition_t>() && traverse_mode_t::RANDOM == opts.mode_) {
    if (
        (traverse_opts_.mode_ == opts.mode_)
        &&
        traverse_range_.size()
    ) {
      return ;  // use cached range
    }
    traverse_opts_ = opts;

    {
      vid_t vertices = traverse_end_ - traverse_start_;
      size_t buckets = (size_t)vertices / CHUNKSIZE + std::min(vertices % CHUNKSIZE, (vid_t)1);

      traverse_range_.resize(buckets);
      vid_t v_start = traverse_start_;

      for (size_t i = 0; i < buckets; ++i) {
        if ((buckets - 1) == i) {
          traverse_range_[i].first  = v_start;
          traverse_range_[i].second = traverse_end_;
        } else {
          traverse_range_[i].first  = v_start;
          traverse_range_[i].second = v_start + CHUNKSIZE;
        }
        v_start = traverse_range_[i].second;
      }
    }

    traverse_i_.store(0, std::memory_order_relaxed);
    std::random_shuffle(traverse_range_.begin(), traverse_range_.end());
  }
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
bool dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::next_chunk(traversal_t traversal, size_t* chunk_size) {
  if (is_seq_part<partition_t>() && traverse_mode_t::RANDOM == traverse_opts_.mode_) {
    vid_t range_start = traverse_i_.fetch_add(*chunk_size, std::memory_order_relaxed);

    if (range_start >= traverse_range_.size()) { return false; }
    if (range_start + *chunk_size > traverse_range_.size()) {
      *chunk_size = traverse_range_.size() - range_start;
    }

    vid_t range_end = range_start + *chunk_size;
    for (vid_t range_i = range_start; range_i < range_end; ++range_i) {
      vid_t v_start = traverse_range_[range_i].first;
      vid_t v_end   = traverse_range_[range_i].second;

      for (vid_t v_i = v_start; v_i < v_end; ++v_i) {
        if (nullptr == traverse_active_ || traverse_active_->get_bit(v_i)) {
          if (false == traversal(v_i, &data_[v_i])) { return true; }
        }
      }
    }
  } else {
    vid_t v_start = traverse_i_.fetch_add(*chunk_size, std::memory_order_relaxed);

    if (v_start >= traverse_end_) { return false; }
    if (v_start + *chunk_size > traverse_end_) { *chunk_size = traverse_end_ - v_start; }

    vid_t v_end = v_start + *chunk_size;

    if (nullptr == traverse_active_) {  // scan all vertices
      if (is_seq_part<partition_t>()) {
        for (vid_t v_i = v_start; v_i < v_end; ++v_i) {
          if (false == traversal(v_i, &data_[v_i])) { return true; }
        }
      } else {  // check vertex in this partition or not
        for (vid_t v_i = v_start; v_i < v_end; ++v_i) {
          if (cluster_info_.partition_id_ != partitioner_->get_partition_id(v_i)) {
            continue;
          }
          if (false == traversal(v_i, &data_[v_i])) { return true; }
        }
      }
    } else {
      const size_t chunk = 64;

      if (is_seq_part<partition_t>()) {
        vid_t v_i = v_start;

        {  // first non-padding chunk
          vid_t __v_end = (v_start + chunk) / chunk * chunk;

          if (__v_end > v_end) { __v_end = v_end; }

          for (; v_i < __v_end; ++v_i) {
            if (0 == traverse_active_->get_bit(v_i)) { continue; }
            if (false == traversal(v_i, &data_[v_i])) { return true; }
          }
        }

        for (; v_i < v_end; v_i += chunk) {
          uint64_t word = traverse_active_->data_[word_offset(v_i)];
          vid_t __v_i = v_i;

          while (word) {
            if (word & 0x01) {
              if (false == traversal(__v_i, &data_[__v_i])) { return true; }
            }

            ++__v_i; if (__v_i >= v_end) { break; }
            word = word >> 1;
          }
        }
      } else {
        vid_t v_i = v_start;

        {  // first non-padding chunk
          vid_t __v_end = (v_start + chunk) / chunk * chunk;

          if (__v_end > v_end) { __v_end = v_end; }

          for (; v_i < __v_end; ++v_i) {
            if (
              (cluster_info_.partition_id_ != partitioner_->get_partition_id(v_i))
                ||
              (0 == traverse_active_->get_bit(v_i))
            ) {
              continue;
            }
            if (false == traversal(v_i, &data_[v_i])) { return true; }
          }
        }

        for (; v_i < v_end; v_i += chunk) {
          uint64_t word = traverse_active_->data_[word_offset(v_i)];
          vid_t __v_i = v_i;

          while (word) {
            if (
              (cluster_info_.partition_id_ == partitioner_->get_partition_id(__v_i))
                &&
              (word & 0x01)
            ) {
              if (false == traversal(__v_i, &data_[__v_i])) { return true; }
            }

            ++__v_i; if (__v_i >= v_end) { break; }
            word = word >> 1;
          }
        }
      }
    }
  }

  return true;
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
template <typename R, typename PROCESS>
R dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::foreach(PROCESS&& process, bitmap_spec_t* pactives, size_t chunk_size) {
  return __foreach<R>(this, std::forward<PROCESS>(process), pactives, chunk_size);
}

// ************************************************************************************ //

}  // namespace plato

#endif

