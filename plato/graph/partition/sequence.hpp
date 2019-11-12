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

#ifndef __PLATO_GRAPH_PARTITION_SEQUENCE_HPP__
#define __PLATO_GRAPH_PARTITION_SEQUENCE_HPP__

#include <cstdint>
#include <cstdlib>

#include <memory>
#include <atomic>
#include <functional>
#include <type_traits>

#include "mpi.h"
#include "glog/logging.h"

#include "plato/graph/base.hpp"
#include "plato/parallel/mpi.hpp"

namespace plato {

namespace {  // helper function

template <typename DT>
void __init_offset(std::vector<vid_t>* poffset, const DT* degrees, vid_t vertices, eid_t edges, int alpha) {
  auto& cluster_info = cluster_info_t::get_instance();

  uint64_t remained_amount = edges + vertices * (uint64_t)alpha;
  uint64_t expected_amount = 0;

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "total_amount: " << remained_amount << ", alpha: " << alpha;
  }

  poffset->clear();
  poffset->resize(cluster_info.partitions_ + 1, 0);
  for (int p_i = 0; p_i < cluster_info.partitions_; ++p_i) {
    expected_amount = remained_amount / (cluster_info.partitions_ - p_i);

    uint64_t amount = 0;
    for (vid_t v_i = poffset->at(p_i); v_i < vertices; ++v_i) {
      amount += (alpha + degrees[v_i]);
      if (amount >= expected_amount) {
        poffset->at(p_i + 1) = v_i / PAGESIZE * PAGESIZE;
        break;
      }
    }
    if ((cluster_info.partitions_ - 1) == p_i) { poffset->at(cluster_info.partitions_) = vertices; }

    remained_amount -= amount;
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "partition-" << p_i << ": [" << poffset->at(p_i) << "," << poffset->at(p_i + 1) << ")"
        << ", amount: " << amount;
    }
  }
}

void __check_consistency(const std::vector<vid_t>& offset_) {
  std::vector<vid_t> offset(offset_.size());

  MPI_Allreduce(offset_.data(), offset.data(), offset_.size(), get_mpi_data_type<vid_t>(),
      MPI_MAX, MPI_COMM_WORLD);

  for (size_t i = 0; i < offset.size(); ++i) {
    CHECK(offset[i] == offset_[i]);
  }
}

}

class sequence_v_view {  // sequence partition view
public:

  // ******************************************************************************* //
  // required types & methods

  // traverse related
  void reset_traversal(const traverse_opts_t& opts = traverse_opts_t());

  /*
   * process a chunk of vertices, thread-safe
   *
   * \tparam TRAVERSAL  traversal functor, it should implement the method:
   *                    <tt>void operator() (vid_t)<\tt>;
   *
   * \return true - traverse at lease one edge, false - no more edges to traverse.
   **/
  template <typename TRAVERSAL>
  bool next_chunk(TRAVERSAL&& traversal, size_t* chunk_size);

  // ******************************************************************************* //

  sequence_v_view(const sequence_v_view&) = delete;
  sequence_v_view& operator=(const sequence_v_view&) = delete;

  sequence_v_view(vid_t start, vid_t end);
  sequence_v_view(sequence_v_view&& other);

  vid_t start() { return start_; }
  vid_t end()   { return end_;   }

protected:
  vid_t              start_;
  vid_t              end_;
  std::atomic<vid_t> traverse_i_;
};

/*
 * Partitioner try to keep each partitions' computation work balanced
 * vertexId must be compacted
 *
 * references:
 *  Julian Shun, Guy E Blelloch. Ligra: A Lightweight Graph Processing
 *  Framework for Shared Memory
 *
 *  Xiaowei Zhu, Wenguang Chen, etc. Gemini: A Computation-Centric Distributed
 *  Graph Processing System
 **/

// edge belong to source node's partition
class sequence_balanced_by_source_t {
public:

  // ******************************************************************************* //
  // required types & methods

  // get edge's partition
  inline int get_partition_id(vid_t src, vid_t /*dst*/) {
    return get_partition_id(src);
  }

  // get vertex's partition
  inline int get_partition_id(vid_t v_i) {
    for (size_t p_i = 0; p_i < (offset_.size() - 1); ++p_i) {
      if (v_i >= offset_[p_i] && v_i < offset_[p_i + 1]) {
        return p_i;
      }
    }
    CHECK(false) << "can not find which partition " << v_i << " belong";
  }

  // get all self vertex's view
  sequence_v_view self_v_view(void) {
    auto& cluster_info = cluster_info_t::get_instance();
    return sequence_v_view(offset_[cluster_info.partition_id_], offset_[cluster_info.partition_id_ + 1]);
  }

  // ******************************************************************************* //

  /*
   * constructor
   *
   * \param degrees   each vertex's degrees
   * \param vertices  vertex number of the graph
   * \param vertices  edge number of the graph
   * \param alpha     vertex's weight of computation, default: -1, means
   *                  alpha = 8 * (partitions - 1)
   **/
  template <typename DT>
  sequence_balanced_by_source_t(const DT* degrees, vid_t vertices, eid_t edges, int alpha = -1) {
    if (-1 == alpha) {
      auto& cluster_info = cluster_info_t::get_instance();
      alpha = 8 * (cluster_info.partitions_ - 1);
    }
    __init_offset(&offset_, degrees, vertices, edges, alpha);
  }

  sequence_balanced_by_source_t(const std::vector<vid_t>& offset): offset_(offset) { }

  void check_consistency(void) { __check_consistency(offset_); }

  // ******************************************************************************* //

  std::vector<vid_t> offset_;

  // ******************************************************************************* //
};

// edge belong to destination node's partition

class sequence_balanced_by_destination_t {
public:

  // ******************************************************************************* //
  // required types & methods

  // get edge's partition
  inline int get_partition_id(vid_t /*src*/, vid_t dst) {
    return get_partition_id(dst);
  }

  // get vertex's partition
  inline int get_partition_id(vid_t v_i) {
    for (size_t p_i = 0; p_i < (offset_.size() - 1); ++p_i) {
      if (v_i >= offset_[p_i] && v_i < offset_[p_i + 1]) {
        return p_i;
      }
    }
    CHECK(false) << "can not find which partition " << v_i << " belong";
  }

  sequence_v_view self_v_view(void) {
    auto& cluster_info = cluster_info_t::get_instance();
    return sequence_v_view(offset_[cluster_info.partition_id_], offset_[cluster_info.partition_id_ + 1]);
  }

  // ******************************************************************************* //

  /*
   * constructor
   *
   * \param degrees   each vertex's degrees
   * \param vertices  vertex number of the graph
   * \param vertices  edge number of the graph
   * \param alpha     vertex's weight of computation, default: -1, means
   *                  alpha = 8 * (partitions - 1)
   **/
  template <typename DT>
  sequence_balanced_by_destination_t(const DT* degrees, vid_t vertices, eid_t edges, int alpha = -1) {
    if (-1 == alpha) {
      auto& cluster_info = cluster_info_t::get_instance();
      alpha = 8 * (cluster_info.partitions_ - 1);
    }
    __init_offset(&offset_, degrees, vertices, edges, alpha);
  }

  sequence_balanced_by_destination_t(const std::vector<vid_t>& offset): offset_(offset) { }

  void check_consistency(void) { __check_consistency(offset_); }

  // ******************************************************************************* //

  std::vector<vid_t> offset_;

  // ******************************************************************************* //
};

template <typename PART>
constexpr bool is_seq_part(void) {
  return std::is_same<PART, sequence_balanced_by_source_t>::value ||
    std::is_same<PART, sequence_balanced_by_destination_t>::value;
}

// ************************************************************************************ //
// implementations

sequence_v_view::sequence_v_view(vid_t start, vid_t end)
  : start_(start), end_(end), traverse_i_(start_) { }

sequence_v_view::sequence_v_view(sequence_v_view&& other)
  : start_(other.start_), end_(other.end_), traverse_i_(start_) { }

void sequence_v_view::reset_traversal(const traverse_opts_t& opts) {
  CHECK(opts.mode_ == traverse_mode_t::ORIGIN);
  traverse_i_.store(start_, std::memory_order_relaxed);
}

template <typename TRAVERSAL>
bool sequence_v_view::next_chunk(TRAVERSAL&& traversal, size_t* chunk_size) {
  vid_t range_start = traverse_i_.fetch_add(*chunk_size, std::memory_order_relaxed);;

  if (range_start >= end_) { return false; }
  if (range_start + *chunk_size > end_) {
    *chunk_size = end_ - range_start;
  }

  vid_t range_end = range_start + *chunk_size;
  for (vid_t range_i = range_start; range_i < range_end; ++range_i) {
    traversal(range_i);
  }
  return true;
}

// ************************************************************************************ //

}  // namespace plato

#endif

