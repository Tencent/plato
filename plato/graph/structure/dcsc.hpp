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

#ifndef __PLATO_GRAPH_STORAGE_DCSC_HPP__
#define __PLATO_GRAPH_STORAGE_DCSC_HPP__

#include <memory>
#include <atomic>
#include <random>
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

/*
 * doubly compressed sparse column storage
 * vertexId must be compacted
 *
 * references:
 *  BULUC, A., AND GILBERT, J.R. On the representation and multiplication of
 *  hypersparse matrices.
 *
 * \tparam EDATA      data type associate with edge
 * \tparam PART_IMPL  partitioner's type
 **/
template <typename EDATA, typename PART_IMPL, typename ALLOC = std::allocator<adj_unit_t<EDATA>>>
class dcsc_t {
private:
  using traits_ = typename std::allocator_traits<ALLOC>::template rebind_traits<adj_unit_t<EDATA>>;

public:

  // ******************************************************************************* //
  // required types & methods

  using edata_t              = EDATA;
  using partition_t          = PART_IMPL;
  using allocator_type       = typename traits_::allocator_type;

  using edge_unit_spec_t     = edge_unit_t<edata_t>;
  using adj_unit_spec_t      = adj_unit_t<edata_t>;
  using adj_unit_list_spec_t = adj_unit_list_t<edata_t>;

  /*
   * load edges from cache
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

  // get partitioner
  std::shared_ptr<partition_t> partitioner(void) { return partitioner_; }

  // traverse interface
  using traversal_t = std::function<bool(vid_t v_i, const adj_unit_list_spec_t&)>;

  /*
   * reset traversal location to start
   **/
  void reset_traversal(traverse_opts_t opts = traverse_opts_t());

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

  dcsc_t(std::shared_ptr<partition_t> partitioner, const allocator_type& alloc = ALLOC());
  dcsc_t(dcsc_t&& other);

  dcsc_t(const dcsc_t&) = delete;

  // build dcsc from exist graph structure
  template <typename GRAPH>
  int load_from_graph(const graph_info_t& graph_info, GRAPH& graph, bool is_outgoing,
      const traverse_opts_t& opts = traverse_opts_t());

  std::shared_ptr<adj_unit_spec_t>         adjs(void)   { return adjs_;   }

  vid_t vertices(void) { return vertices_; }
  eid_t edges(void)    { return edges_;    }

protected:

  using adjs_allocator_t = typename traits_::template rebind_alloc<adj_unit_spec_t>;
  using adjs_traits_     = typename traits_::template rebind_traits<adj_unit_spec_t>;
  using adjs_pointer     = typename adjs_traits_::pointer;

  using index_allocator_t = typename traits_::template rebind_alloc<eid_t>;
  using index_traits_     = typename traits_::template rebind_traits<eid_t>;
  using index_pointer     = typename index_traits_::pointer;


  using local_vertex_allocator_t = typename traits_::template rebind_alloc<vid_t>;
  using local_vertex_traits_     = typename traits_::template rebind_traits<vid_t>;
  using local_vertex_pointer     = typename local_vertex_traits_::pointer;

  vid_t vertices_;
  eid_t edges_;

  std::shared_ptr<partition_t>             partitioner_;
  std::shared_ptr<adj_unit_spec_t>         adjs_;
  std::shared_ptr<eid_t> index_;
  std::shared_ptr<vid_t> local_vertex_;

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
  int load_from_traversal(vid_t vertices, std::function<void(bool)> reset_traversal,
      std::function<void(bsp_send_callback_t<vid_t>)> foreach_dests,
      std::function<void(bsp_send_callback_t<edge_unit_spec_t>)> foreach_edges);
};

// ************************************************************************************ //
// implementations

template <typename EDATA, typename PART_IMPL, typename ALLOC>
dcsc_t<EDATA, PART_IMPL, ALLOC>::dcsc_t(std::shared_ptr<partition_t> partitioner, const allocator_type& alloc)
  : vertices_(0), edges_(0), partitioner_(partitioner), allocator_(alloc), traverse_i_(0) { }

template <typename EDATA, typename PART_IMPL, typename ALLOC>
dcsc_t<EDATA, PART_IMPL, ALLOC>::dcsc_t(dcsc_t&& other)
  : vertices_(other.vertices_), edges_(other.edges_),
    partitioner_(std::move(other.partitioner_)), adjs_(std::move(other.adjs_)), index_(std::move(other.index_)),
    local_vertex_(std::move(other.local_vertex_)), allocator_(other.allocator_), traverse_i_(0), 
    traverse_range_(std::move(other.traverse_range_)), traverse_opts_(other.traverse_opts_) { }

template <typename EDATA, typename PART_IMPL, typename ALLOC>
int dcsc_t<EDATA, PART_IMPL, ALLOC>::load_from_traversal(vid_t vertices, std::function<void(bool)> reset_traversal,
    std::function<void(bsp_send_callback_t<vid_t>)> foreach_dests,
    std::function<void(bsp_send_callback_t<edge_unit_spec_t>)> foreach_edges) {

  int rc = -1;
  bsp_opts_t opts;
  auto& cluster_info = cluster_info_t::get_instance();

  // build compressed index
  {
    vid_t tmp_vertices = vertices;
    index_allocator_t __alloc(allocator_);
    auto* __p = __alloc.allocate(vertices + 1);
    memset(__p, 0, sizeof(eid_t) * (vertices + 1));

    index_.reset(__p, [__alloc, tmp_vertices](index_pointer p) mutable {
      __alloc.deallocate(p, tmp_vertices + 1);
    });
  }

  {  // count each vertex's in degree
    bitmap_t<> v_bitmap(vertices);

    opts.threads_               = -1;
    opts.flying_send_per_node_  = 3;
    opts.flying_recv_           = cluster_info.partitions_;
    opts.global_size_           = 16 * MBYTES;
    opts.local_capacity_        = PAGESIZE;
    opts.batch_size_            = 1;

    auto __send = [&](bsp_send_callback_t<vid_t> send) {
      foreach_dests(send);
    };

    auto __recv = [&](int /*p_i*/, bsp_recv_pmsg_t<vid_t>& pmsg) {
      __sync_fetch_and_add(&index_.get()[*pmsg + 1], (eid_t)1);
      v_bitmap.set_bit(*pmsg);
    };

    reset_traversal(false);
    if (0 != (rc = fine_grain_bsp<vid_t>(__send, __recv, opts))) {
      LOG(ERROR) << "bsp failed with code: " << rc;
      return -1;
    }

    vertices_ = v_bitmap.count();
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "[staging-1]: count vertex's in-degree done";
  }

  for (size_t i = 1; i <= vertices; ++i) {
    index_.get()[i] = index_.get()[i] + index_.get()[i - 1];
  }

  edges_ = index_.get()[vertices];
  LOG(INFO) << "[staging-1]: [" << cluster_info.partition_id_ << "] local edges(" << edges_ << ")"
    << ", local vertices(" << vertices_ << ")";

  {
    eid_t tmp_edges = edges_;

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

  {  // store edges
    opts.threads_               = -1;
    opts.flying_send_per_node_  = 3;
    opts.flying_recv_           = cluster_info.partitions_;
    opts.global_size_           = 16 * MBYTES;
    opts.local_capacity_        = 32 * PAGESIZE;
    opts.batch_size_            = 1;

    auto __send = [&](bsp_send_callback_t<edge_unit_spec_t> send) {
      foreach_edges(send);
    };

    auto __recv = [&](int /*p_i*/, bsp_recv_pmsg_t<edge_unit_spec_t>& pmsg) {
      eid_t idx = __sync_fetch_and_add(&index_.get()[pmsg->dst_], (eid_t)1);

      auto& nei = adjs_.get()[idx];
      nei.neighbour_ = pmsg->src_;
      nei.edata_     = pmsg->edata_;
    };

    reset_traversal(true);
    if (0 != (rc = fine_grain_bsp<edge_unit_spec_t>(__send, __recv, opts))) {
      LOG(ERROR) << "bsp failed with code: " << rc;
      return -1;
    }
    for (size_t i = vertices - 1; i >= 1; --i) {
      index_.get()[i] = index_.get()[i - 1];
    }
    index_.get()[0] = 0;
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "[staging-2]: store edge done.";
  }

  {
    vid_t tmp_vertices = vertices_;
    local_vertex_allocator_t __alloc(allocator_);
    auto* __p = __alloc.allocate(vertices_ + 1);
    memset(__p, 0, sizeof(vid_t) * (vertices_ + 1));

    local_vertex_.reset(__p, [__alloc, tmp_vertices](local_vertex_pointer p) mutable {
      __alloc.deallocate(p, tmp_vertices + 1);
    });

  }

  {
    vid_t idx = 0;
    for (vid_t v_i = 0; v_i < vertices; ++v_i) {
      if (0 != (index_.get()[v_i + 1] - index_.get()[v_i])) {
        index_.get()[idx] = index_.get()[v_i];
        local_vertex_.get()[idx] = v_i;
        ++idx;
      }
    }

    CHECK(idx == vertices_);
    index_.get()[vertices_]= index_.get()[vertices];
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "[staging-3]: build compressed index done.";
  }

  return 0;
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
template <typename EDGE_CACHE>
int dcsc_t<EDATA, PART_IMPL, ALLOC>::load_from_cache(const graph_info_t& graph_info, EDGE_CACHE& cache) {
  auto reset_traversal = [&](bool auto_release_ = false) {
    traverse_opts_t opts; opts.mode_ = traverse_mode_t::RANDOM;
    opts.auto_release_ = auto_release_;
    cache.reset_traversal(opts);
  };

  auto foreach_dests = [&](bsp_send_callback_t<vid_t> send) {
    auto traversal = [&](size_t, edge_unit_spec_t* edge) {
      CHECK(edge->src_ < graph_info.vertices_);
      CHECK(edge->dst_ < graph_info.vertices_);

      send(partitioner_->get_partition_id(edge->src_, edge->dst_), edge->dst_);
      if (false == graph_info.is_directed_) {  // cache friendly
        send(partitioner_->get_partition_id(edge->dst_, edge->src_), edge->src_);
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
      if (false == graph_info.is_directed_) {  // cache friendly
        vid_t tmp = edge->src_; edge->src_ = edge->dst_; edge->dst_ = tmp;
        send(partitioner_->get_partition_id(edge->src_, edge->dst_), *edge);
      }
      return true;
    };

    size_t chunk_size = 64;
    while (cache.next_chunk(traversal, &chunk_size)) {
    }
  };

  return load_from_traversal(graph_info.vertices_, reset_traversal, foreach_dests, foreach_edges);
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
template <typename GRAPH>
int dcsc_t<EDATA, PART_IMPL, ALLOC>::load_from_graph(const graph_info_t& graph_info, GRAPH& graph, bool is_outgoing,
    const traverse_opts_t& opts) {
  auto reset_traversal = [&](bool) { graph.reset_traversal(opts); };

  auto foreach_dests = [&](bsp_send_callback_t<vid_t> send) {
    auto traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
      if (is_outgoing) {
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          send(partitioner_->get_partition_id(v_i, it->neighbour_), it->neighbour_);
        }
      } else {
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          send(partitioner_->get_partition_id(it->neighbour_, v_i), v_i);
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

  return load_from_traversal(graph_info.vertices_, reset_traversal, foreach_dests, foreach_edges);
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
void dcsc_t<EDATA, PART_IMPL, ALLOC>::reset_traversal(traverse_opts_t opts) {
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
    size_t buckets = std::min((size_t)(MBYTES), (size_t)vertices_ / basic_chunk + 1);

    traverse_range_.resize(buckets);
    vid_t remained_vertices = vertices_;
    vid_t v_start = 0;

    for (size_t i = 0; i < buckets; ++i) {
      if ((buckets - 1) == i) {
        traverse_range_[i].first  = v_start;
        traverse_range_[i].second = vertices_;
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
      int   p_i     = 0;
      vid_t v_i     = 0;
      vid_t p_v_end = partition_end<partition_t>(p_i);
      std::vector<std::pair<vid_t, vid_t>> p_ranges(cluster_info.partitions_, std::make_pair(0, 0));

      // divid & conquer to speed up ??
      for (; v_i < vertices_ && p_i < cluster_info.partitions_; ++v_i) {
        if (local_vertex_.get()[v_i] >= p_v_end) {
          p_ranges[p_i].second = v_i;

          ++p_i;
          if (p_i < cluster_info.partitions_) {
            p_ranges[p_i].first = v_i;
            p_v_end = partition_end<partition_t>(p_i);
          }
        }
      }
      if (p_i < cluster_info.partitions_) {
        p_ranges[p_i].second = vertices_;
      }

      p_i = (cluster_info.partition_id_ + 1) % cluster_info.partitions_;
      do {
        uint64_t v_start = p_ranges[p_i].first;
        uint64_t v_end   = p_ranges[p_i].second;

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

#ifdef __DCSC_DEBUG__
  if (0 == cluster_info_t::get_instance().partition_id_) {
    for (size_t i = 0; i < traverse_range_.size(); ++i) {
      LOG(INFO) << "traverse_range_[" << i << "]: " << traverse_range_[i].first << ", " << traverse_range_[i].second;
    }
  }
#endif
}

template <typename EDATA, typename PART_IMPL, typename ALLOC>
bool dcsc_t<EDATA, PART_IMPL, ALLOC>::next_chunk(traversal_t traversal, size_t* chunk_size) {
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
      eid_t idx_start = index_.get()[v_i];
      eid_t idx_end   = index_.get()[v_i + 1];

      if (false == traversal(local_vertex_.get()[v_i], adj_unit_list_spec_t(&adjs_.get()[idx_start], &adjs_.get()[idx_end]))) {
        break;
      }
    }
  }

  return true;
}

// ************************************************************************************ //

}

#endif

