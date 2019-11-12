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

#ifndef __PLATO_ALGO_KCORE_HPP__
#define __PLATO_ALGO_KCORE_HPP__

#include <cstdlib>
#include <cstdint>

#include <limits>
#include <vector>
#include <algorithm>
#include <functional>
#include <type_traits>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"

#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"

namespace plato { namespace algo{

struct kcore_info_t {
  plato::vid_t v_count_;
  plato::eid_t e_count_;
};

enum class kcore_calc_type_t {
  SUBGRAPH = 1,
  VERTEX   = 2
};

class kcore_algo_t {
public:

  /*
   * Finds the coreness (shell index) of the vertices of the network.
   * Ref: Montresor A, De Pellegrini F, Miorandi D. Distributed k-core decomposition[J].
   *  IEEE Transactions on parallel and distributed systems, 2012, 24(2): 288-300.
   *
   * \tparam Graph        The graph type, the graph must be partition by destination node
   * \tparam CALLBACK     The type of callback function
   *
   * \param  graph_info   Graph infomations
   * \param  incomings    Graph edges indexed by destination node
   * \param  callback     The result callback function
   **/
  template <typename Graph, typename Callback>
  static dense_state_t<vid_t, typename Graph::partition_t>
  compute_shell_index(const graph_info_t& graph_info, Graph& incomings,
      Callback&& callback);
};

template <typename Graph, typename Callback>
dense_state_t<vid_t, typename Graph::partition_t>
kcore_algo_t::compute_shell_index(const graph_info_t& graph_info, Graph& incomings, Callback&& callback) {
  using partition_t   = typename Graph::partition_t;
  using adj_unit_list_spec_t = typename Graph::adj_unit_list_spec_t;
  using bitmap_spec_t = bitmap_t<>;
  using state_t       = dense_state_t<vid_t, partition_t>;

  constexpr bool is_seq = std::is_same<partition_t, sequence_balanced_by_destination_t>::value
    || std::is_same<partition_t, sequence_balanced_by_source_t>::value;
  static_assert(is_seq, "kcore only support sequence partition now");

  plato::stop_watch_t watch;
  watch.mark("t0");
  watch.mark("t1");
  auto& cluster_info = cluster_info_t::get_instance();

  bitmap_spec_t active_current(graph_info.max_v_i_ + 1);
  bitmap_spec_t active_next(graph_info.max_v_i_ + 1);
  state_t       estimate(graph_info.max_v_i_, incomings.partitioner());
  state_t       coreness(graph_info.max_v_i_, incomings.partitioner());

  active_current.fill();
  size_t need_modified = graph_info.vertices_;
  size_t modified = graph_info.vertices_;
  coreness.fill(std::numeric_limits<vid_t>::max());

  // init coreness with node's degree
  incomings.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 64;
    while (incomings.next_chunk([&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
      coreness[v_i] = adjs.end_ - adjs.begin_;
      return true;
    }, &chunk_size)) { }
  }

  int partitions    = cluster_info.partitions_;
  int partition_id  = cluster_info.partition_id_;
  vid_t avg_degrees = std::max(graph_info.edges_ / graph_info.vertices_, 1UL);
  auto& offsets     = incomings.partitioner()->offset_;

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "prepared for shell-index caculation done, cost: " << watch.showlit_seconds("t1");
  }

  std::vector<int> displs(partitions);
  std::vector<int> counts(partitions);

  for (int p_i = 0; p_i < partitions; ++p_i) {
    counts[p_i] = offsets[p_i + 1] - offsets[p_i];
    displs[p_i] = offsets[p_i];
  }
  CHECK(displs[partitions - 1] >= 0) << "Allgatherv exceed int32 limit. displs[partitions-1]: "
    << displs[partitions - 1];

  int epoch = 0;
  do {
    struct broadcast_msg_t {
      vid_t v_i;
      vid_t coreness;
    };

    watch.mark("t1");
    watch.mark("t2");

    std::shared_ptr<bitmap_spec_t> active_wrapper(&active_current, [](bitmap_spec_t*) { });

    size_t lowbound = graph_info.vertices_ / avg_degrees / (sizeof(broadcast_msg_t) / sizeof(vid_t));
    bool is_broadcast_sparse = (modified < lowbound / 8)
      || ((sizeof(vid_t) * modified > 500 * MBYTES) && (modified < lowbound));
    if (is_broadcast_sparse) {  // sparse mode
      using broadcast_ctx_t = mepa_bc_context_t<broadcast_msg_t>;
      auto active_view = create_active_v_view(incomings.partitioner()->self_v_view(), active_current);
      broadcast_message<broadcast_msg_t, vid_t>(active_view,
        [&](const broadcast_ctx_t& ctx, vid_t v_i) {
          ctx.send(broadcast_msg_t { v_i, coreness[v_i] });
        },
        [&](int, broadcast_msg_t& msg) {
          estimate[msg.v_i] = msg.coreness;
          return 0;
        });
    } else {  // dense mode
      // MPI_IN_PLACE is slow for MPI_Allgatherv
      int rc = MPI_Allgatherv(&coreness[offsets[partition_id]], offsets[partition_id + 1] - offsets[partition_id],
          get_mpi_data_type<vid_t>(), &estimate[0], counts.data(), displs.data(),
          get_mpi_data_type<vid_t>(), MPI_COMM_WORLD);
      CHECK_EQ(MPI_SUCCESS, rc);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    double broadcast_cost = watch.show("t2");
    watch.mark("t2");

    // update coreness
    modified = 0;
    need_modified = 0;
    active_next.clear();

    vid_t actives = 0;

    // XXX(ced) Here we do not flow montresor'12' algo(update node's coreness until no update can be performed)
    // if communication is a problem, refactor here.
    actives = 0;
    coreness.reset_traversal(active_wrapper);
    #pragma omp parallel reduction(+:actives)
    {
      size_t chunk_size = 64;
      vid_t __actives = 0;
      while (coreness.next_chunk([&](vid_t v_i, vid_t* pcrns) {
        static thread_local std::vector<vid_t> __count;

        // caculate h-index
        auto adjs     = incomings.neighbours(v_i);
        vid_t adjcnt  = adjs.end_ - adjs.begin_;

        if (0 == adjcnt) {
          *pcrns = 0;
          ++__actives;
          return true;
        }

        vid_t est = std::min(adjcnt, estimate[v_i]);
        __count.assign(est + 1, 0);

        for (auto it = adjs.begin_; it != adjs.end_; ++it) {
          ++__count[std::min(est, estimate[it->neighbour_])];
        }

        vid_t sum = 0;
        for (vid_t i = est; i > 0; --i) {
          sum += __count[i];

          if (sum >= i) {
            if (i < *pcrns) {
              *pcrns = i;
              for (auto it = adjs.begin_; it != adjs.end_; ++it) {
                if (!active_next.get_bit(it->neighbour_)) {  // get_bit is not a atomic ops
                  active_next.set_bit(it->neighbour_);
                }
              }
              ++__actives;
              __atomic_fetch_add(&modified, 1, __ATOMIC_RELAXED);
            }
            break;
          }
        }
        return true;
      }, &chunk_size)) { }
      actives += __actives;
    }
    MPI_Allreduce(MPI_IN_PLACE, &modified, 1, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);

    if (actives) {
      coreness.reset_traversal(active_wrapper);
      #pragma omp parallel
      {
        size_t chunk_size = 64;
        while (coreness.next_chunk([&](vid_t v_i, vid_t* pcrns) {
          estimate[v_i] = *pcrns;
          return true;
        }, &chunk_size)) { }
      }
    }

    active_next.sync();
    need_modified = active_next.count();

    ++epoch;
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "epoch: " << epoch
        << ", modified: " << modified
        << ", need_modified: " << need_modified
        << ", broadcast[" << is_broadcast_sparse << "] cost: " << broadcast_cost << "ms"
        << ", update cost: " << watch.showlit_seconds("t2")
        << ", one epoch cost: " << watch.showlit_seconds("t1");
    }
    std::swap(active_next, active_current);
  } while (need_modified);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "caculation done, cost: " << watch.showlit_seconds("t0")
      << ", now saving result to hdfs...";
  }
  return coreness;
}

}} // plato algo

#endif

