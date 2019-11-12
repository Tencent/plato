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
#include <list>
#include <vector>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "boost/align.hpp"
#include "boost/format.hpp"

#include "plato/graph/base.hpp"
#include "plato/graph/state/sparse_state.hpp"
#include "plato/util/perf.hpp"
#include "plato/util/defer.hpp"
#include "plato/util/hash.hpp"
#include "plato/util/bitmap.hpp"
#include "plato/parallel/mpi.hpp"
#include "plato/parallel/bsp.hpp"
#include "plato/graph/partition/hash.hpp"

namespace plato {

#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

template <typename EDATA, typename UDATA>
struct tcsr_state_t {
  adj_unit_t<EDATA>* adjs_;
  vid_t adjs_size_;
  UDATA data_;

  tcsr_state_t(const tcsr_state_t&) = delete;
  tcsr_state_t& operator=(const tcsr_state_t&) = delete;
  tcsr_state_t(tcsr_state_t&& x) = default;
  tcsr_state_t& operator=(tcsr_state_t&& x) = default;

  tcsr_state_t(adj_unit_t<EDATA>* adjs) noexcept : adjs_(adjs), adjs_size_(0) { }
};

struct mmap_deleter {
  size_t _size;
  void operator()(void* ptr) const {
    ::munmap(ptr, _size);
  }
};

/*
 * table-based compressed sparse adjacent storage
 * use it when vertexId is sparse
 *
 * \tparam EDATA      data type associate with edge
 * \tparam VDATA      data type associate with vertex
 * \tparam PART_IMPL  partitioner's type
 * \tparam HASH       a unary function object that hash vid_t to size_t
 * \tparam KEY_EQUAL  a binary predicate that takes two arguments of the key type and returns a bool
 * \tparam ALLOC      type of the allocator object used to define the storage allocation model
 * \tparam BITMAP     bitmap used for traverse vertex's state
 *
 **/
template <typename EDATA, typename VDATA, typename PART_IMPL, typename HASH = cuckoo_vid_hash,
  typename KEY_EQUAL = std::equal_to<vid_t>,
  typename ALLOC = std::allocator<std::pair<const vid_t, tcsr_state_t<EDATA, VDATA>>>,
  typename BITMAP = bitmap_t<>>
class tcsr_t {
public:

  // ******************************************************************************* //
  // required types & methods

  using edata_t              = EDATA;
  using partition_t          = PART_IMPL;

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
  int load_edges_from_cache(
    const graph_info_t& graph_info, EDGE_CACHE& cache, std::unique_ptr<vid_t, mmap_deleter>&& out_degree, size_t degree_sum);

  template <typename EDGE_CACHE>
  int load_edges_from_cache(const graph_info_t& graph_info, EDGE_CACHE& cache);

  // query interface
  inline adj_unit_list_spec_t neighbours(vid_t v_i);

  // get partitioner
  std::shared_ptr<partition_t> partitioner(void) { return data_.partitioner(); }

  /*
   * reset traversal location to start
   *
   **/
  void reset_traversal(const traverse_opts_t& = traverse_opts_t()) { data_.reset_traversal(); }

  /*
   * process a chunk of edges, thread-safe
   *
   * \param traversal    callback function: void(vid_t v_i, const adj_unit_list_spec_t&)
   * \param chunk_size   input, traverse at most chunk_size vertices,
   *                     output, real vertices traversed
   *
   * \return
   *    true  -- traverse at lease one edge
   *    false -- no more edges to traverse
   **/
  template <typename TraversalCallback>
  bool next_chunk(TraversalCallback&& traversal, size_t* chunk_size);

  // ******************************************************************************* //

  // TODO we should move these function to state class ??

  using u_data_t       = VDATA;
  using graph_data_t   = tcsr_state_t<EDATA, VDATA>;
  using state_t        = sparse_state_t<graph_data_t, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>;
  using adj_unit_vec_t = std::vector<adj_unit_t<edata_t>>;

  // returns a reference to the value belong to v_i, v_i must be existed
  graph_data_t&       operator[] (vid_t v_i)       { return data_[v_i]; }
  const graph_data_t& operator[] (vid_t v_i) const { return data_[v_i]; }

  // return underlying storage
  state_t& data() { return data_; }

  /*
   * load vertex's state from cache
   *
   * \tparam cache             vertex state cache
   * \tparam update_callback   void update_callback(vid_t, VDATA&, V_MEMBER_T&)
   *
   **/
  template <typename V_MEMBER_T, typename VERTICES_CACHE, typename UpdateCallback>
  int load_vertices_data_from_cache(VERTICES_CACHE& cache, UpdateCallback&& update_callback);

protected:
  state_t data_;
  std::unique_ptr<adj_unit_spec_t, mmap_deleter> adjs_;
public:
  tcsr_t(size_t n, std::shared_ptr<partition_t> partition, const HASH& hfunc = HASH(),
      const KEY_EQUAL& equal = KEY_EQUAL(), const ALLOC& alloc = ALLOC())
    : data_(n, std::move(partition), hfunc, equal, alloc) { }

  tcsr_t(tcsr_t&& other) noexcept : data_(std::move(other.data_)), adjs_(std::move(other.adjs_)) { }
  tcsr_t& operator=(tcsr_t &&other) noexcept {
    if (this != &other) {
      this->~tcsr_t();
      new(this) tcsr_t(std::move(other));
    }
    return *this;
  }

  tcsr_t(const tcsr_t&) = delete;
  tcsr_t& operator=(const tcsr_t&) = delete;
};

// ************************************************************************************ //
// implementations

template <typename EDATA, typename VDATA, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
adj_unit_list_t<EDATA> tcsr_t<EDATA, VDATA, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::neighbours(vid_t v_i) {
  const graph_data_t& v = data_[v_i];
  adj_unit_list_spec_t neis;
  neis.begin_ = v.adjs_;
  neis.end_ = v.adjs_ + v.adjs_size_;
  return neis;
}

template <typename EDATA, typename VDATA, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
template <typename TraversalCallback>
bool tcsr_t<EDATA, VDATA, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::next_chunk(
    TraversalCallback&& traversal, size_t* chunk_size) {
  return data_.next_chunk([&](vid_t v_i, graph_data_t* pv) {
    adj_unit_list_spec_t neis;
    neis.begin_ = pv->adjs_;
    neis.end_   = pv->adjs_ + pv->adjs_size_;
    traversal(v_i, neis);
    return true;
  }, chunk_size);
}

template <typename EDATA, typename VDATA, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
template <typename EDGE_CACHE>
int tcsr_t<EDATA, VDATA, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::load_edges_from_cache(
  const graph_info_t& graph_info, EDGE_CACHE& cache, std::unique_ptr<vid_t, mmap_deleter>&& out_degree, size_t degree_sum) {
  auto& cluster_info = cluster_info_t::get_instance();
  auto& partition = *data_.partitioner();

  size_t mem_size = sizeof(adj_unit_spec_t) * degree_sum;
  adjs_ = std::unique_ptr<adj_unit_spec_t, mmap_deleter>(
    (adj_unit_spec_t*)mmap(nullptr, mem_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
    mmap_deleter{mem_size});
  CHECK(MAP_FAILED != adjs_.get())
  << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno) << " degree_sum: " << degree_sum;

  bsp_opts_t opts;
  opts.threads_               = -1;
  opts.flying_send_per_node_  = 2;
  opts.flying_recv_           = std::min(cluster_info.partitions_, cluster_info.threads_);
  opts.global_size_           = 32 * MBYTES;
  opts.local_capacity_        = 4 * PAGESIZE;
  opts.batch_size_            = 1;

  plato::stop_watch_t watch;

  watch.mark("t0");
  {
    watch.mark("t1");

    data_.unlock();
    auto lock_defer = defer([&]{ data_.lock(); });

    size_t adj_offset = 0;
    #pragma omp parallel for
    for (size_t v_i = 0; v_i <= graph_info.max_v_i_; ++v_i) {
      vid_t degree = *(out_degree.get() + v_i);
      if (partition.get_partition_id(v_i) == cluster_info.partition_id_ && degree) {
        data_.upsert(v_i, [] (graph_data_t&) { CHECK(false) << "duplicated vertex!"; }, adjs_.get() + __sync_fetch_and_add(&adj_offset, degree));
      }
    }

    CHECK(adj_offset == degree_sum);
    out_degree.reset();
    MPI_Barrier(MPI_COMM_WORLD);
    LOG_IF(INFO, 0 == cluster_info.partition_id_) << "build index only table cost: " << watch.show("t1") / 1000.0 << "s";
  }

  {
    watch.mark("t1");
    cache.reset_traversal();
    auto __send = [&] (bsp_send_callback_t<edge_unit_spec_t> send) {
      auto traversal = [&](size_t, const edge_unit_spec_t * edge) {
        send(partition.get_partition_id(edge->src_, edge->dst_), *edge);
        if (!graph_info.is_directed_) {
          edge_unit_spec_t edge_swap;
          edge_swap.src_ = edge->dst_;
          edge_swap.dst_ = edge->src_;
          edge_swap.edata_ = edge->edata_;
          send(partition.get_partition_id(edge_swap.src_, edge_swap.dst_), edge_swap);
        }
        return true;
      };

      size_t chunk_size = 64;
      while (cache.next_chunk(traversal, &chunk_size)) { }
    };

    auto __recv = [&] (int /*p_i*/, bsp_recv_pmsg_t<edge_unit_spec_t>& pmsg) {
      edge_unit_spec_t& msg = *pmsg;
      graph_data_t& data = data_[msg.src_];
      vid_t index = __sync_fetch_and_add(&data.adjs_size_, 1);
      adj_unit_spec_t& adj = data.adjs_[index];
      adj.neighbour_ = msg.dst_;
      adj.edata_ = msg.edata_;
    };

    auto rc = fine_grain_bsp<edge_unit_spec_t>(__send, __recv, opts);
    if (0 != rc) {
      LOG(ERROR) << "bsp failed with code: " << rc;
      return -1;
    }

    size_t adj_num = 0;
    data_.reset_traversal();
    #pragma omp parallel reduction(+:adj_num)
    {
      size_t chunk_size = 64;
      while (data_.next_chunk([&] (vid_t /* v_i */, graph_data_t* data) {
        CHECK(data->adjs_size_);
        adj_num += data->adjs_size_;
      }, &chunk_size)) { }
    }
    CHECK(adj_num == degree_sum) << "adj_num: " << adj_num << ", degree_sum: " << degree_sum;

    LOG_IF(INFO, 0 == cluster_info.partition_id_) << "fill table cost: " << watch.show("t1") / 1000.0 << "s";
  }

  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "build table cost total: " << watch.show("t0") / 1000.0 << "s";
  return 0;
}

template <typename EDATA, typename VDATA, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
template <typename EDGE_CACHE>
int tcsr_t<EDATA, VDATA, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::load_edges_from_cache(
  const graph_info_t& graph_info, EDGE_CACHE& cache) {
  auto& cluster_info = cluster_info_t::get_instance();
  auto& partition = *data_.partitioner();

  plato::stop_watch_t watch;
  watch.mark("t0");

  size_t out_degree_mem_size = (boost::alignment::align_up(sizeof(vid_t) * (size_t(graph_info.max_v_i_) + 1), PAGESIZE));
  std::unique_ptr<vid_t, mmap_deleter> out_degree(
    (vid_t*)mmap(nullptr, out_degree_mem_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
    mmap_deleter{out_degree_mem_size});
  CHECK(MAP_FAILED != out_degree.get())
  << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno);

  cache.reset_traversal();

  #pragma omp parallel
  {
    size_t chunk_size = 64;
    while (cache.next_chunk(
      [&out_degree, &graph_info] (size_t, edge_unit_spec_t* edge) {
        __sync_fetch_and_add(out_degree.get() + edge->src_, 1);
        if (!graph_info.is_directed_) {
          __sync_fetch_and_add(out_degree.get() + edge->dst_, 1);
        }
        return true;
      },
      &chunk_size
    )) { }
  }

  size_t chunk_size = 128 * 1024 * 1024;
  size_t total_size = size_t(graph_info.max_v_i_) + 1;
  for (size_t i = 0; i < total_size; i += chunk_size) {
    MPI_Allreduce(MPI_IN_PLACE, out_degree.get() + i, std::min(chunk_size, total_size - i), get_mpi_data_type<vid_t>(), MPI_SUM, MPI_COMM_WORLD);
  }
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "count out_degree cost: " << watch.show("t0") / 1000.0 << "s";

  size_t local_degree_sum = 0;
  #pragma omp parallel for reduction(+:local_degree_sum)
  for (size_t v_i = 0; v_i <= graph_info.max_v_i_; v_i++) {
    if (partition.get_partition_id(v_i) == cluster_info.partition_id_) {
      local_degree_sum += *(out_degree.get() + v_i);
    }
  }

  return load_edges_from_cache(graph_info, cache, std::move(out_degree), local_degree_sum);
}

template <typename EDATA, typename VDATA, typename PART_IMPL, typename HASH, typename KEY_EQUAL, typename ALLOC, typename BITMAP>
template <typename V_MEMBER_T, typename VERTICES_CACHE, typename UpdateCallback>
int tcsr_t<EDATA, VDATA, PART_IMPL, HASH, KEY_EQUAL, ALLOC, BITMAP>::load_vertices_data_from_cache(
    VERTICES_CACHE& cache, UpdateCallback&& update_callback) {
  auto& partition = *data_.partitioner();

  data_.unlock();
  auto lock_defer = defer([&] { data_.lock(); });
  UNUSED(lock_defer);

  bsp_opts_t opts;
  auto& cluster_info = cluster_info_t::get_instance();

  {
    opts.threads_               = -1;
    opts.flying_send_per_node_  = std::max((int)cluster_info.threads_ / cluster_info.partitions_, 3);
    opts.flying_recv_           = cluster_info.partitions_;
    opts.global_size_           = 64 * MBYTES;
    opts.local_capacity_        = 32 * PAGESIZE;
    opts.batch_size_            = 1;

    traverse_opts_t traverse_opts;
    traverse_opts.mode_ = traverse_mode_t::RANDOM;
    cache.reset_traversal(traverse_opts);

    auto __send = [&](bsp_send_callback_t<vertex_unit_t<V_MEMBER_T>> send) {
      size_t chunk_size = 64;
      while (cache.next_chunk([&](size_t, vertex_unit_t<V_MEMBER_T>* v_data) {
          send(partition.get_partition_id(v_data->vid_), *v_data);
          return true;
        }, &chunk_size)) {}
    };

    auto __recv = [&] (int /* p_i */, bsp_recv_pmsg_t<vertex_unit_t<V_MEMBER_T>>& pmsg) {
      CHECK(data_.update(pmsg->vid_, [&] (graph_data_t& v_data) {
        update_callback(pmsg->vid_, v_data.data_, pmsg->vdata_);
      })) << "non vertex find!";
    };

    auto rc = fine_grain_bsp<vertex_unit_t<V_MEMBER_T>>(__send, __recv, opts);
    if (0 != rc) {
      LOG(ERROR) << "bsp failed with code: " << rc;
      return -1;
    }
  }

  return 0;
}

}  // namespace plato

