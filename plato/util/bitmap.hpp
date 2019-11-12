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

#ifndef __PLATO_UTIL_BITMAP_HPP__
#define __PLATO_UTIL_BITMAP_HPP__

#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include "mpi.h"

#include <vector>
#include <functional>
#include <type_traits>

#include "plato/parallel/mpi.hpp"
#include "plato/graph/base.hpp"

namespace plato {

inline size_t word_offset(size_t idx) { return (idx >> 6); };
inline size_t bits_offset(size_t idx) { return (idx & (size_t)0x3F); };

template <typename ALLOC = std::allocator<uint64_t>>
class bitmap_t {
public:
  using traits_        = typename std::allocator_traits<ALLOC>::template rebind_traits<uint64_t>;
  using allocator_type = typename traits_::allocator_type;

  bitmap_t(const allocator_type& alloc = ALLOC());
  bitmap_t(size_t size, const allocator_type& alloc = ALLOC());

  bitmap_t(bitmap_t&&);
  bitmap_t& operator=(bitmap_t&&);

  bitmap_t(const bitmap_t&) = delete;

  ~bitmap_t(void);

  int copy_from(const bitmap_t& other);

  void clear(void); // clear this bitmap
  void fill(void);  // set all bits to 1

  inline void     set_bit(size_t i);
  inline void     clr_bit(size_t i);
  inline uint64_t get_bit(size_t i) const ;

  size_t msb(void)   const ; // find most significant bit of this bitmap
  size_t count(void) const ; // count non-zero bits of this bitmap
  void   sync(void) ;         // sync all bitmap in this cluster
  void   allreduce_band(void) ;
  std::vector<size_t> to_vector(void);

  // traverse related

  using traversal_t = std::function<void(vid_t)>;

  /*
   * reset traversal location to start
   **/
  void reset_traversal(const traverse_opts_t& opts = traverse_opts_t()) const ;

  /*
   * process a chunk of veteices, thread-safe
   *
   * \param traversal    callback function to deal with 
   * \param chunk_size   input, traverse at most chunk_size * 64 vertices,
   *                     output, real vertices traversed
   *
   * \return
   *    true  -- traverse at lease one vertex
   *    false -- no more vertex to traverse
   * */
  bool next_chunk(traversal_t traversal, size_t* chunk_size) const ;

  size_t              size_;
  uint64_t*           data_;
  allocator_type      allocator_;

protected:

  // traverse related
  static const vid_t basic_chunk = 64;
  mutable std::atomic<size_t>                  traverse_i_;
  mutable std::vector<std::pair<vid_t, vid_t>> traverse_range_;
  mutable traverse_opts_t                      traverse_opts_;

};

// ************************************************************************************ //
// implementations

template <typename ALLOC>
bitmap_t<ALLOC>::bitmap_t(const allocator_type& alloc)
  : size_(0), data_(nullptr), allocator_(alloc), traverse_i_(0) { }

template <typename ALLOC>
bitmap_t<ALLOC>::bitmap_t(size_t size, const allocator_type& alloc)
  : size_(size), data_(nullptr), allocator_(alloc), traverse_i_(0) {
  data_ = allocator_.allocate(word_offset(size_) + 1);
  clear();
}

template <typename ALLOC>
bitmap_t<ALLOC>::bitmap_t(bitmap_t&& other)
  : size_(other.size_), data_(other.data_), allocator_(other.allocator_),
    traverse_i_(0) {
  other.data_ = nullptr;
  other.size_ = 0;
}

template <typename ALLOC>
bitmap_t<ALLOC>& bitmap_t<ALLOC>::operator=(bitmap_t&& other) {
  if (nullptr != data_) {
    allocator_.deallocate(data_, word_offset(size_) + 1);
  }

  size_      = other.size_;
  data_      = other.data_;
  allocator_ = other.allocator_;

  other.size_ = 0;
  other.data_ = nullptr;

  traverse_i_.store(other.traverse_i_);
  traverse_range_ = std::move(other.traverse_range_);
  traverse_opts_  = other.traverse_opts_;

  return *this;
}

template <typename ALLOC>
bitmap_t<ALLOC>::~bitmap_t(void) {
  if (nullptr != data_) {
    allocator_.deallocate(data_, word_offset(size_) + 1);
  }
}

template <typename ALLOC>
void bitmap_t<ALLOC>::clear(void) {
  size_t bm_size = word_offset(size_);
  #pragma omp parallel for
  for (size_t i = 0; i <= bm_size; ++i) {
    data_[i] = 0;
  }
}

template <typename ALLOC>
void bitmap_t<ALLOC>::fill(void) {
  size_t bm_size = word_offset(size_);
  #pragma omp parallel for
  for (size_t i = 0; i < bm_size; ++i) {
    data_[i] = 0xffffffffffffffffUL;
  }
  data_[bm_size] = 0;
  for (size_t i = (bm_size << 6); i < size_; ++i) {
    data_[bm_size] |= 1ul << bits_offset(i);
  }
}

template <typename ALLOC>
size_t bitmap_t<ALLOC>::msb(void) const {
  size_t i = word_offset(size_);
  do {
    if (data_[i]) {
      uint64_t word = data_[i];
      uint64_t msb  = 0;

      while (word) {
        word = word / 2;
        ++msb;
      }

      return i * 64UL + msb - 1;
    }
  } while(i--);
  return 0;
}

template <typename ALLOC>
int bitmap_t<ALLOC>::copy_from(const bitmap_t& other) {
  if (size_ != other.size_) {
    return -1;
  }

  size_t bm_size = word_offset(size_);
  #pragma omp parallel for
  for (size_t i = 0; i <= bm_size; ++i) {
    data_[i] = other.data_[i];
  }
  return 0;
}

template <typename ALLOC>
uint64_t bitmap_t<ALLOC>::get_bit(size_t i) const {
  return data_[word_offset(i)] & (1ul << bits_offset(i));
}

template <typename ALLOC>
void bitmap_t<ALLOC>::set_bit(size_t i) {
  __sync_fetch_and_or(data_ + word_offset(i), 1ul << bits_offset(i));
}

template <typename ALLOC>
void bitmap_t<ALLOC>::clr_bit(size_t i) {
  __sync_fetch_and_and(data_ + word_offset(i), ~(1ul << bits_offset(i)));
}

template <typename ALLOC>
size_t bitmap_t<ALLOC>::count(void) const {
  size_t cnt = 0;
  size_t bm_size = word_offset(size_);
  #pragma omp parallel for reduction(+:cnt)
  for (size_t i = 0; i <= bm_size; ++i) {
    cnt += __builtin_popcountl(data_[i]);
  }
  return cnt;
}

template <typename ALLOC>
void bitmap_t<ALLOC>::sync(void) {
  MPI_Allreduce(MPI_IN_PLACE, data_, word_offset(size_) + 1, get_mpi_data_type<uint64_t>(), MPI_BOR, MPI_COMM_WORLD);
}

template <typename ALLOC>
void bitmap_t<ALLOC>::allreduce_band() {
  MPI_Allreduce(MPI_IN_PLACE, data_, word_offset(size_) + 1, get_mpi_data_type<uint64_t>(), MPI_BAND, MPI_COMM_WORLD);
}
template <typename ALLOC>
std::vector<size_t> bitmap_t<ALLOC>::to_vector(void) {
  std::vector<size_t> vec;
  size_t bm_size = word_offset(size_);
  for (size_t i = 0; i <= bm_size; ++i) {
    if (data_[i]) {
      for (size_t b_i = 0; b_i < 64; ++b_i) {
        if (data_[i] & (1UL << b_i)) {
          vec.emplace_back(i * 64 + b_i);
        }
      }
    }
  }
  return vec;
}

template <typename ALLOC>
void bitmap_t<ALLOC>::reset_traversal(const traverse_opts_t& opts) const {
  traverse_i_.store(0, std::memory_order_relaxed);

  if (
      (traverse_opts_.mode_ == opts.mode_)
      &&
      traverse_range_.size()
  ) {
    return ;  // use cached range
  }
  traverse_opts_ = opts;

  auto build_ranges_origin = [&](void) {
    size_t total_blocks = word_offset(size_) + 1;
    size_t buckets      = std::min((size_t)(MBYTES), total_blocks / basic_chunk + 1);

    traverse_range_.resize(buckets);
    vid_t remained_blocks = (vid_t)total_blocks;
    vid_t v_start = 0;

    for (size_t i = 0; i < buckets; ++i) {
      if ((buckets - 1) == i) {
        traverse_range_[i].first  = v_start;
        traverse_range_[i].second = (vid_t)total_blocks;
      } else {
        size_t amount = remained_blocks / (buckets - i) / basic_chunk * basic_chunk;

        traverse_range_[i].first  = v_start;
        traverse_range_[i].second = v_start + (vid_t)amount;
      }

      v_start = traverse_range_[i].second;
      remained_blocks -= (traverse_range_[i].second - traverse_range_[i].first);
    }
    CHECK(0 == remained_blocks);
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

#ifdef __BITMAP_DEBUG__
  if (0 == cluster_info_t::get_instance().partition_id_) {
    for (size_t i = 0; i < traverse_range_.size(); ++i) {
      LOG(INFO) << "traverse_range_[" << i << "]: " << traverse_range_[i].first << ", " << traverse_range_[i].second;
    }
  }
#endif
}

template <typename ALLOC>
bool bitmap_t<ALLOC>::next_chunk(traversal_t traversal, size_t* chunk_size) const {
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
      uint64_t word = data_[v_i];

      for (int b_i = 0; 0 != word; word >>= 1, ++b_i) {
        if (word & 0x01) {
          traversal(v_i * 64 + b_i);
        }
      }
    }
  }

  return true;
}

// ************************************************************************************ //

}  // namespace plato

#endif
