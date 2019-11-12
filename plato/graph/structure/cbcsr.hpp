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

#ifndef __PLATO_GRAPH_STORAGE_CBCSR_HPP__
#define __PLATO_GRAPH_STORAGE_CBCSR_HPP__

#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>

#include <cstdint>
#include <cstring>

#include <limits>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <utility>
#include <algorithm>
#include <functional>
#include <type_traits>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "plato/graph/base.hpp"
#include "plato/graph/partition/sequence.hpp"
#include "plato/util/bitmap.hpp"
#include "plato/util/perf.hpp"
#include "plato/util/mmap_alloc.hpp"
#include "plato/parallel/mpi.hpp"
#include "plato/parallel/bsp.hpp"

namespace plato {

/**
 * @brief alias_block_t
 */
struct alias_block_t {
  float  prob_;
  plato::eid_t alias_;
};

/**
 * @brief graph storage type: compressed bcsr_t
 * @tparam EDATA - edge data type
 * @tparam PART_IMPL - partition implication
 * @tparam ALLOC - memory allocator
 */
template <typename EDATA, typename PART_IMPL, typename ALLOC = std::allocator<adj_unit_t<EDATA>>>
class cbcsr_t {
private:
  using traits_ = typename std::allocator_traits<ALLOC>::template rebind_traits<adj_unit_t<EDATA>>;

public:

  using bitmap_spec_t  = bitmap_t<ALLOC>;

  // ******************************************************************************* //
  // required types & methods

  using edata_t              = EDATA;
  using partition_t          = PART_IMPL;
  using allocator_type       = typename traits_::allocator_type;

  using edge_unit_spec_t     = edge_unit_t<edata_t>;
  using adj_unit_spec_t      = adj_unit_t<edata_t>;
  using adj_unit_list_spec_t = adj_unit_list_t<edata_t>;

  /*
   * load edges from iterator
   *
   * \tparam EDGE_CACHE  edges' cache type
   *
   * \param  graph_info  vertices & edges must be supplied
   * \param  cache       edges' cache
   *
   * \return 0 -- success, else failed
   **/
  template <typename EDGE_CACHE>
  int load_from_cache(const graph_info_t& graph_info, EDGE_CACHE& cache);

  // Bitmap-Assist-CSR does not support dynamic graph
  //
  // int add_edge(const edge_unit_spec_t& edge) { return 0; }
  // int add_edges(const std::vector<edge_unit_spec_t>& edges) { return 0; }

  inline adj_unit_list_spec_t neighbours(vid_t v_i);

  // get partitioner
  std::shared_ptr<partition_t> partitioner(void) { return partitioner_; }

  // traverse interface
  using traversal_t = std::function<bool(vid_t v_i, const adj_unit_list_spec_t&)>;

  /*
   * reset traversal location to start
   *
   **/
  void reset_traversal(const traverse_opts_t& opts = traverse_opts_t());

  /*
   * process a chunk of edges, thread-safe
   *
   * \param traversal    callback function to deal with
   * \param chunk_size   input, traverse at most chunk_size vertices,
   *                     output, real vertices traversed
   *
   * \return
   *    true  -- traverse at lease one edge
   *    false -- no more edges to traverse
   * */
  bool next_chunk(traversal_t traversal, size_t* chunk_size);

  // ******************************************************************************* //

  /**
   * @brief cbcsr_t constructor
   * @param partitioner - vertex partitioner
   * @param alloc - memory allocator
   */
  cbcsr_t(std::shared_ptr<PART_IMPL> partitioner, const allocator_type& alloc = ALLOC());

  /**
   * @brief cbcsr_t move constructor
   * @param other
   */
  cbcsr_t(cbcsr_t&& other);

  /**
   * @brief cbcsr_t no copy constructor
   */
  cbcsr_t(const cbcsr_t&) = delete;
  cbcsr_t& operator=(const cbcsr_t&) = delete;

  /**
   * @brief cbcsr_t from graph
   * @tparam GRAPH - graph storage
   * @param graph_info
   * @param graph
   * @param is_outgoing
   * @param opts - option
   * @return
   */
  template <typename GRAPH>
  int load_from_graph(
    const graph_info_t& graph_info, GRAPH& graph, bool is_outgoing,
    const traverse_opts_t& opts = traverse_opts_t());

  /**
   * @brief bitmap getter
   * @return bitmap_ ref
   */
  std::shared_ptr<bitmap_spec_t>   bitmap(void) { return bitmap_; }

  /**
   * @brief adjs getter
   * @return adjs_ ref
   */
  std::shared_ptr<adj_unit_spec_t> adjs(void)   { return adjs_;   }

  /**
   * @brief index getter
   * @return index_
   */
  std::shared_ptr<eid_t>           index(void)  { return index_;  }

  /**
   * @brief vertices getter
   * @return vertices_ ref
   */
  vid_t vertices(void)       { return vertices_;      }

  /**
   * @brief edges getter
   * @return edges_ ref
   */
  eid_t edges(void)          { return edges_;         }

  /**
   * @brief non_zero_lines getter
   * @return non_zero_lines_ ref
   */
  vid_t non_zero_lines(void) { return non_zero_lines_; }

  /**
   * @brief max_vid getter
   * @return max_vid_ ref
   */
  vid_t max_vid(void)        { return max_vid_;        }

  /**
   * @brief get_random_edge
   * @tparam URNG
   * @param v_i - vertex
   * @param g
   * @return adj_unit_spec_t* - edge
   */
  template <typename URNG>
  adj_unit_spec_t* get_random_edge(vid_t v_i, URNG& g);

  /**
   * @brief
   * @param v_i
   * @param target
   * @return
   */
  bool existed(vid_t v_i, vid_t target);

protected:

  using bitmap_allocator_t = typename traits_::template rebind_alloc<bitmap_spec_t>;
  using bitmap_traits_     = typename traits_::template rebind_traits<bitmap_spec_t>;
  using bitmap_pointer     = typename bitmap_traits_::pointer;

  using buckets_allocator_t = typename traits_::template rebind_alloc<vid_t>;
  using buckets_traits_     = typename traits_::template rebind_traits<vid_t>;
  using buckets_pointer     = typename buckets_traits_::pointer;

  using rows_allocator_t    = typename traits_::template rebind_alloc<vid_t>;
  using rows_traits_        = typename traits_::template rebind_traits<vid_t>;
  using rows_pointer        = typename rows_traits_::pointer;

  using adjs_allocator_t = typename traits_::template rebind_alloc<adj_unit_spec_t>;
  using adjs_traits_     = typename traits_::template rebind_traits<adj_unit_spec_t>;
  using adjs_pointer     = typename adjs_traits_::pointer;

  using index_allocator_t = typename traits_::template rebind_alloc<eid_t>;
  using index_traits_     = typename traits_::template rebind_traits<eid_t>;
  using index_pointer     = typename index_traits_::pointer;

  using alias_allocator_t = typename traits_::template rebind_alloc<alias_block_t>;
  using alias_traits_     = typename traits_::template rebind_traits<alias_block_t>;
  using alias_pointer     = typename alias_traits_::pointer;

  vid_t vertices_;
  vid_t max_vid_;
  eid_t edges_;
  std::shared_ptr<PART_IMPL>        partitioner_;
  vid_t bucket_size_;
  vid_t non_zero_lines_;

  std::shared_ptr<bitmap_spec_t>    bitmap_;
  std::shared_ptr<adj_unit_spec_t>  adjs_;
  std::shared_ptr<eid_t>            index_;
  std::shared_ptr<vid_t>            buckets_;
  std::shared_ptr<vid_t>            rows_;
  std::shared_ptr<alias_block_t>    alias_table_;
  allocator_type allocator_;

  // traverse related
  static const vid_t basic_chunk = 64;  // do not basic_chunk too big, try to keep parallel edges access within L3 cache

  std::atomic<vid_t>                   traverse_i_;
  std::vector<std::pair<vid_t, vid_t>> traverse_range_;
  traverse_opts_t                      traverse_opts_;

  // SFINAE only works for deduced template arguments, it's tricky here
  template <typename PART>
  typename std::enable_if<is_seq_part<PART>(), vid_t>::type partition_start(int p_i) {
    return partitioner_->offset_[p_i];
  }
  template <typename PART>
  typename std::enable_if<is_seq_part<PART>(), vid_t>::type partition_end(int p_i) {
    return partitioner_->offset_[p_i + 1];
  }

  // SFINAE only works for deduced template arguments, it's tricky here
  template <typename PART>
  typename std::enable_if<!is_seq_part<PART>(), vid_t>::type partition_start(int) {
    CHECK(false); return -1;
  }
  template <typename PART>
  typename std::enable_if<!is_seq_part<PART>(), vid_t>::type partition_end(int) {
    CHECK(false); return -1;
  }

  // helper function for load edges
  int load_from_traversal(
    vid_t vertices, vid_t max_vid, std::function<void(bool)> reset_traversal,
    std::function<void(bsp_send_callback_t<vid_t>)> foreach_srcs,
    std::function<void(bsp_send_callback_t<edge_unit_spec_t>)> foreach_edges);
};

template <typename EDATA, typename PART_IMPL, typename ALLOC>
cbcsr_t<EDATA, PART_IMPL, ALLOC>::cbcsr_t(std::shared_ptr<PART_IMPL> partitioner, const allocator_type& alloc)
  : vertices_(0), max_vid_(0), edges_(0), partitioner_(partitioner),
    allocator_(alloc), traverse_i_(0) { }

template <typename EDATA, typename PART_IMPL, typename ALLOC>
cbcsr_t<EDATA, PART_IMPL, ALLOC>::cbcsr_t(cbcsr_t&& other)
  : vertices_(other.vertices_), edges_(other.edges_), max_vid_(other.max_v_i_), partitioner_(std::move(other.partitioner_)),
    bitmap_(std::move(other.bitmap_)), adjs_(std::move(other.adjs_)), index_(std::move(other.index_)), buckets_(std::move(other.buckets_)), rows_(std::move(other.rows_)),
    alias_table_(std::move(other.alias_table_)), allocator_(other.allocator_), traverse_i_(0), traverse_range_(std::move(other.traverse_range_)),
    traverse_opts_(other.traverse_opts_) { }

template <typename EDATA, typename PART_IMPL, typename ALLOC>
int cbcsr_t<EDATA, PART_IMPL, ALLOC>::load_from_traversal(
  vid_t vertices, vid_t max_vid, std::function<void(bool)> reset_traversal,
  std::function<void(bsp_send_callback_t<vid_t>)> foreach_srcs,
  std::function<void(bsp_send_callback_t<edge_unit_spec_t>)> foreach_edges) {

  plato::stop_watch_t watch;
  auto& cluster_info = cluster_info_t::get_instance();
  eid_t tmp_edges    = 0;
  vertices_ = vertices;
  max_vid_ = max_vid;
  watch.mark("t4");
  {
    bitmap_allocator_t __alloc(allocator_);
    auto* __p = __alloc.allocate(1);
    __alloc.construct(__p, max_vid_ + 1);

    bitmap_.reset(__p, [__alloc](bitmap_pointer p) mutable {
      __alloc.destroy(p);
      __alloc.deallocate(p, 1);
    });
  }
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "malloc bitmap----> " << watch.show("t4") / 1000.0 << "s";
  }
  int rc = -1;
  bsp_opts_t opts;


  {  // init bitmap
    opts.threads_               = -1;
    opts.flying_send_per_node_  = 3;
    opts.flying_recv_           = cluster_info.partitions_;
    opts.global_size_           = 64 * MBYTES;
    opts.local_capacity_        = 32 * PAGESIZE;
    opts.batch_size_            = 1;

    traverse_opts_t trvs_opts; trvs_opts.mode_ = traverse_mode_t::RANDOM;
    reset_traversal(false);

    edges_ = 0;

    auto __send = [&](bsp_send_callback_t<vid_t> send) {
      foreach_srcs(send);
    };

    auto __recv = [&](int /*p_i*/, bsp_recv_pmsg_t<vid_t>& pmsg) {
      bitmap_->set_bit(*pmsg);
    };
    watch.mark("t0");
    if (0 != (rc = fine_grain_bsp<vid_t>(__send, __recv, opts))) {
      LOG(ERROR) << "bsp failed with code: " << rc;
      return -1;
    }
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "partition id: " << cluster_info.partition_id_ << " >> bsp vertex----> " << watch.show("t0") / 1000.0 << "s";
    LOG(INFO) << "[staging-1]: initial bitmap done";
  }

  vid_t non_zero_lines = bitmap_->count();
  non_zero_lines_ = non_zero_lines;
  bucket_size_ = (max_vid_ + 1 + non_zero_lines) / non_zero_lines;  //Set the bucket size to total rows divided by non-zero rows
  vid_t bucket_num = (vid_t)(((uint64_t)max_vid_ + 1UL + bucket_size_) / bucket_size_); //The number of buckets
  vid_t tmp_bucket_num = bucket_num;
  vid_t tmp_non_zero_lines = non_zero_lines;
  watch.mark("t5");
  {
    buckets_allocator_t __alloc(allocator_);
    auto* __p = __alloc.allocate(bucket_num+1);
    memset(__p, 0, sizeof(vid_t) * (bucket_num + 1));

    buckets_.reset(__p, [__alloc, tmp_bucket_num](buckets_pointer p) mutable {
      __alloc.deallocate(p, tmp_bucket_num + 1);
    });
  }

  {
    rows_allocator_t __alloc(allocator_);
    auto* __p = __alloc.allocate(non_zero_lines);
    memset(__p, 0, sizeof(vid_t) * (non_zero_lines));

    rows_.reset(__p, [__alloc, tmp_non_zero_lines](rows_pointer p) mutable {
      __alloc.deallocate(p, tmp_non_zero_lines);
    });
  }

  tmp_non_zero_lines = non_zero_lines;
  {
    index_allocator_t __alloc(allocator_);
    auto* __p = __alloc.allocate(non_zero_lines + 1);
    memset(__p, 0, sizeof(eid_t) * (non_zero_lines + 1));

    index_.reset(__p, [__alloc, tmp_non_zero_lines](index_pointer p) mutable {
      __alloc.deallocate(p, tmp_non_zero_lines + 1);
    });
  }
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "malloc buckets & rows & index----> " << watch.show("t5") / 1000.0 << "s";
  }
  watch.mark("t1");
  vid_t idx = 0;
  size_t bm_size = plato::word_offset(max_vid_ + 1);
  for (size_t i = 0; i <= bm_size; ++i) {
    if (bitmap_->data_[i]) {
      for (size_t b_i = 0; b_i < 64; ++b_i) {
        if (bitmap_->data_[i] & (1UL << b_i)) {
          vid_t vtx = (i * 64 + b_i);
          vid_t my_bucket = vtx / bucket_size_ ;
          ++buckets_.get()[my_bucket + 1];
          rows_.get()[idx++] = vtx;
        }
      }
    }
  }
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "Initialize the buckets & rows times----> " << watch.show("t1") / 1000.0 << "s";
  }

  watch.mark("t7");
  for(unsigned i = 1; i <= bucket_num; ++i) {
    buckets_.get()[i] = buckets_.get()[i] + buckets_.get()[i - 1];
  }
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "cal the buckets----> " << watch.show("t7") / 1000.0 << "s";
  }

  {  // count each vertex's out degree
    opts.threads_               = -1;
    opts.flying_send_per_node_  = 3;
    opts.flying_recv_           = cluster_info.partitions_;
    opts.global_size_           = 64 * MBYTES;
    opts.local_capacity_        = 32 * PAGESIZE;
    opts.batch_size_            = 1;

    traverse_opts_t trvs_opts; trvs_opts.mode_ = traverse_mode_t::RANDOM;
    reset_traversal(false);

    auto __send = [&](bsp_send_callback_t<vid_t> send) {
      foreach_srcs(send);
    };

    auto __recv = [&](int /*p_i*/, bsp_recv_pmsg_t<vid_t>& pmsg) {
      vid_t v_i = *pmsg;
      vid_t my_bucket = v_i / bucket_size_;
      vid_t start = buckets_.get()[my_bucket];
      vid_t end = buckets_.get()[my_bucket + 1];
      auto pt = std::lower_bound(rows_.get() + start, rows_.get() + end, v_i);
      CHECK(pt != rows_.get() + end);
      vid_t pos = pt - rows_.get();
      CHECK(rows_.get()[pos] == v_i);
      __sync_fetch_and_add(&index_.get()[pos + 1], 1);

    };

    watch.mark("t2");
    if (0 != (rc = fine_grain_bsp<vid_t>(__send, __recv, opts))) {
      LOG(ERROR) << "bsp failed with code: " << rc;
      return -1;
    }
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "The second time the data is synchronized----> " << watch.show("t2") / 1000.0 << "s";
    LOG(INFO) << "[staging-2]: count vertex's out-degree done";
  }
  watch.mark("t8");
  for(size_t i = 1; i <= non_zero_lines; ++i) {
    index_.get()[i] = index_.get()[i] + index_.get()[i - 1];
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "cal the index----> " << watch.show("t8") / 1000.0 << "s";
  }
  edges_ = index_.get()[non_zero_lines];

  tmp_edges = edges_;
  watch.mark("t6");
  {
    mmap_allocator_t<adj_unit_spec_t> __alloc; //use mmap noreserve
    auto* __p = __alloc.allocate(edges_);
    if (false == std::is_trivial<adj_unit_spec_t>::value) {
      for (size_t i = 0; i < edges_; ++i) {
        __alloc.construct(&__p[i]);
      }
    }
    adjs_.reset(__p, [__alloc, tmp_edges](adjs_pointer p) mutable {
      if (false == std::is_trivial<adj_unit_spec_t>::value) {
        for (size_t i = 0; i < tmp_edges; ++i) {
          __alloc.destroy(&p[i]);
        }
      }
      __alloc.deallocate(p, tmp_edges);
    });
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "malloc the adjs----> " << watch.show("t6") / 1000.0 << "s";
  }

  {  // store edges
    opts.threads_               = -1;
    opts.flying_send_per_node_  = 3;
    opts.flying_recv_           = cluster_info.partitions_;
    opts.global_size_           = 16 * MBYTES;
    opts.local_capacity_        = 32 * PAGESIZE;
    opts.batch_size_            = 1;

    traverse_opts_t trvs_opts; trvs_opts.mode_ = traverse_mode_t::RANDOM;
    reset_traversal(true);

    auto __send = [&](bsp_send_callback_t<edge_unit_spec_t> send) {
      foreach_edges(send);
    };

    auto __recv = [&](int /*p_i*/, bsp_recv_pmsg_t<edge_unit_spec_t>& pmsg) {
      vid_t v_i = pmsg->src_;
      vid_t my_bucket = v_i / bucket_size_;
      vid_t start = buckets_.get()[my_bucket];
      vid_t end = buckets_.get()[my_bucket + 1];
      auto pt = std::lower_bound(rows_.get() + start, rows_.get() + end, v_i);
      CHECK(pt != rows_.get() + end);
      vid_t pos = pt - rows_.get();
      CHECK(rows_.get()[pos] == v_i);
      eid_t idx = __sync_fetch_and_add(&index_.get()[pos], (eid_t)1);

      auto& nei = adjs_.get()[idx];
      nei.neighbour_ = pmsg->dst_;
      nei.edata_     = pmsg->edata_;
    };

    watch.mark("t3");
    if (0 != (rc = fine_grain_bsp<edge_unit_spec_t>(__send, __recv, opts))) {
      LOG(ERROR) << "bsp failed with code: " << rc;
      return -1;
    }

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "The third time the data is synchronized----> " << watch.show("t3") / 1000.0 << "s";
      LOG(INFO) << "[staging-2]: count vertex's out-degree done";
    }

    for (size_t i = non_zero_lines - 1; i >= 1; --i) {
      index_.get()[i] = index_.get()[i - 1];
    }

    index_.get()[0] = 0;
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "[staging-3]: store edge done.";
  }

  return 0;
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
typename cbcsr_t<EDATA, PART_IMPL, ALLOC>::adj_unit_list_spec_t cbcsr_t<EDATA, PART_IMPL, ALLOC>::neighbours(vid_t v_i) {

  adj_unit_list_spec_t neis;
  if (bitmap_->get_bit(v_i)) {
    vid_t pos = max_vid_ + 1;
    vid_t my_bucket = v_i / bucket_size_;
    vid_t start = buckets_.get()[my_bucket];
    vid_t end = buckets_.get()[my_bucket + 1];
    auto pt = std::lower_bound(rows_.get() + start, rows_.get() + end, v_i);
    CHECK(pt != rows_.get() + end);
    pos = pt - rows_.get();
    CHECK(rows_.get()[pos] == v_i);

    eid_t start_i = index_.get()[pos];
    eid_t end_i   = index_.get()[pos + 1];
    neis.begin_ = &adjs_.get()[start_i];
    neis.end_   = &adjs_.get()[end_i];
  } else {
    neis.end_ = neis.begin_ = nullptr;
  }
  return neis;
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
template <typename URNG>
typename cbcsr_t<EDATA, PART_IMPL, ALLOC>::adj_unit_spec_t* cbcsr_t<EDATA, PART_IMPL, ALLOC>::get_random_edge(vid_t v_i, URNG& g) {
  auto neis = this->neighbours(v_i);
  if((neis.begin_ == NULL && neis.end_ == NULL) || neis.begin_ == neis.end_) {
    return NULL;
  }

  size_t neighbour_count = neis.end_ - neis.begin_;
  std::uniform_int_distribution<vid_t> dist(0, neighbour_count - 1);
  return (neis.begin_ + dist(g));
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
bool cbcsr_t<EDATA, PART_IMPL, ALLOC>::existed(vid_t v_i, vid_t target) {
  auto neis = this->neighbours(v_i);
  if((neis.begin_ == NULL && neis.end_ == NULL) || neis.begin_ == neis.end_) {
    return false;
  }

  auto* pt = neis.begin_;
  while(pt != neis.end_) {
    if(pt->neighbour_ == target) {
      return true;
    }
    ++pt;
  }

  return false;
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
template <typename EDGE_CACHE>
int cbcsr_t<EDATA, PART_IMPL, ALLOC>::load_from_cache(const graph_info_t& graph_info, EDGE_CACHE& cache) {
  auto reset_traversal = [&](bool auto_release_ = false) {
    traverse_opts_t trvs_opts; trvs_opts.mode_ = traverse_mode_t::RANDOM;
    trvs_opts.auto_release_ = auto_release_;
    cache.reset_traversal(trvs_opts);
  };

  auto foreach_srcs = [&](bsp_send_callback_t<vid_t> send) {
    auto traversal = [&](size_t, edge_unit_spec_t* edge) {

      send(partitioner_->get_partition_id(edge->src_, edge->dst_), edge->src_);
      if (false == graph_info.is_directed_) {
        send(partitioner_->get_partition_id(edge->dst_, edge->src_), edge->dst_);
      }
      return true;
    };

    size_t chunk_size = 64;
    while (cache.next_chunk(traversal, &chunk_size)) {
    }
  };

  auto foreach_edges = [&](bsp_send_callback_t<edge_unit_spec_t> send) {
    auto traversal = [&](size_t, edge_unit_spec_t* edge) {
      send(partitioner_->get_partition_id(edge->src_, edge->dst_), *edge);
      if (false == graph_info.is_directed_) {
        auto tmp = edge->src_; edge->src_ = edge->dst_; edge->dst_ = tmp;
        send(partitioner_->get_partition_id(edge->src_, edge->dst_), *edge);
      }
      return true;
    };

    size_t chunk_size = 64;
    while (cache.next_chunk(traversal, &chunk_size)) {
    }
  };
  return load_from_traversal(graph_info.vertices_, graph_info.max_v_i_, reset_traversal, foreach_srcs, foreach_edges);
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
template <typename GRAPH>
int cbcsr_t<EDATA, PART_IMPL, ALLOC>::load_from_graph(
  const graph_info_t& graph_info, GRAPH& graph, bool is_outgoing,
  const traverse_opts_t& opts) {
  auto reset_traversal = [&](bool) { graph.reset_traversal(opts); };

  auto foreach_srcs = [&](bsp_send_callback_t<vid_t> send) {
    auto traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
      if (is_outgoing) {
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          send(partitioner_->get_partition_id(v_i, it->neighbour_), v_i);
        }
      } else {
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          send(partitioner_->get_partition_id(it->neighbour_, v_i), it->neighbour_);
        }
      }
      return true;
    };
    size_t chunk_size = 1;
    while (graph.next_chunk(traversal, &chunk_size)) { }
  };

  auto foreach_edges = [&](bsp_send_callback_t<edge_unit_spec_t> send) {
    auto traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
      for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
        edge_unit_spec_t edge;
        if (is_outgoing) {
          edge.src_ = v_i; edge.dst_ = it->neighbour_; edge.edata_ = it->edata_;
        } else {
          edge.dst_ = v_i; edge.src_ = it->neighbour_; edge.edata_ = it->edata_;
        }
        send(partitioner_->get_partition_id(edge.src_, edge.dst_), edge);
      }
      return true;
    };
    size_t chunk_size = 1;
    while (graph.next_chunk(traversal, &chunk_size)) { }
  };

  return load_from_traversal(graph_info.vertices_, graph_info.max_v_i_, reset_traversal, foreach_srcs, foreach_edges);
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
void cbcsr_t<EDATA, PART_IMPL, ALLOC>::reset_traversal(const traverse_opts_t& opts) {
  traverse_i_.store(0, std::memory_order_release);

  if (
    (traverse_opts_.mode_ == opts.mode_)
    &&
    traverse_range_.size()
    ) {
    return ;  // use cached range
  }
  traverse_opts_ = opts;

  auto build_ranges_origin = [&](void) {
    size_t buckets = std::min((size_t)(MBYTES), (size_t)non_zero_lines_ / basic_chunk + 1);

    traverse_range_.resize(buckets);
    vid_t remained_vertices = non_zero_lines_;
    vid_t v_start = 0;

    for (size_t i = 0; i < buckets; ++i) {
      if ((buckets - 1) == i) {
        traverse_range_[i].first  = v_start;
        traverse_range_[i].second = non_zero_lines_;
      } else {
        size_t amount = remained_vertices / (buckets - i) / basic_chunk * basic_chunk;

        traverse_range_[i].first  = v_start;
        traverse_range_[i].second = v_start + amount;
      }

      v_start = traverse_range_[i].second;
      remained_vertices -= (traverse_range_[i].second - traverse_range_[i].first);
    }
    CHECK(0 == remained_vertices);
  };

  switch (traverse_opts_.mode_) {
    case traverse_mode_t::ORIGIN:
      build_ranges_origin();
      break;
    case traverse_mode_t::RANDOM:
      build_ranges_origin();
      std::random_shuffle(traverse_range_.begin(), traverse_range_.end());
      break;
    case traverse_mode_t::CIRCLE:
    {
      auto& cluster_info = cluster_info_t::get_instance();
      traverse_range_.clear();

      if (is_seq_part<partition_t>()) {
        int p_i = (cluster_info.partition_id_ + 1) % cluster_info.partitions_;

        do {
          uint64_t v_start = partition_start<partition_t>(p_i);
          uint64_t v_end   = partition_end<partition_t>(p_i);

          for (vid_t v_i = v_start; v_i < v_end; v_i += basic_chunk) {
            uint64_t __v_end = v_i + basic_chunk > v_end ? v_end : v_i + basic_chunk;
            traverse_range_.emplace_back(std::make_pair((vid_t)v_i, (vid_t)__v_end));
          }
          p_i = (p_i + 1) % cluster_info.partitions_;
        } while (p_i != ((cluster_info.partition_id_ + 1) % cluster_info.partitions_));
      } else {
        CHECK(false) << "only sequence partition support circular traverse";
      }
      break;
    }
    default:
      CHECK(false) << "unknown/unsupported traverse mode!";
  }
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
bool cbcsr_t<EDATA, PART_IMPL, ALLOC>::next_chunk(traversal_t traversal, size_t* chunk_size) {
  vid_t range_start = traverse_i_.fetch_add(*chunk_size, std::memory_order_relaxed);

  if (range_start >= traverse_range_.size()) { return false; }
  if (range_start + *chunk_size > traverse_range_.size()) {
    *chunk_size = traverse_range_.size() - range_start;
  }

  vid_t range_end = range_start + *chunk_size;
  for (vid_t range_i = range_start; range_i < range_end; ++range_i) {
    vid_t v_start = traverse_range_[range_i].first;
    vid_t v_end   = traverse_range_[range_i].second;

    for (vid_t v_idx = v_start; v_idx < v_end; ++v_idx) {

      eid_t idx_start = index_.get()[v_idx];
      eid_t idx_end   = index_.get()[v_idx + 1];

      if (false == traversal(rows_.get()[v_idx], adj_unit_list_spec_t(&adjs_.get()[idx_start], &adjs_.get()[idx_end]))) {
        break;
      }
    }
  }

  return true;
}

}
#endif