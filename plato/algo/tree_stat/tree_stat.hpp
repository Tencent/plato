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

#ifndef __PLATO_ALGO_TREESTAT_HPP__
#define __PLATO_ALGO_TREESTAT_HPP__

#include <cstdint>
#include <cstdlib>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"

namespace plato { namespace algo {

struct tree_stat_t {
  uint32_t width_;
  uint32_t depth_;
  bool is_tree_;
};

/*
 * demo implementation of tree-stat
 *
 * \tparam INCOMING   graph type, with incoming edges
 * \tparam OUTGOING   graph type, with outgoing edges
 *
 * \param in_edges    incoming edges, dcsc, ...
 * \param out_edges   outgoing edges, bcsr, ...
 * \param graph_info  base graph-info
 * \param root        id of root
 *
 * \return
 *    tree_stat_t
 * */

template <typename INCOMING, typename OUTGOING>
tree_stat_t tree_stat(
    INCOMING& in_edges,
    OUTGOING& out_edges,
    const graph_info_t& graph_info,
    const plato::vid_t root) {

  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();
  tree_stat_t stat {0, 0, true};
  
  dualmode_engine_t<INCOMING, OUTGOING> engine (
    std::shared_ptr<INCOMING>(&in_edges,  [](INCOMING*) { }),
    std::shared_ptr<OUTGOING>(&out_edges, [](OUTGOING*) { }),
    graph_info);

  plato::vid_t actives = 1;

  auto visited        = engine.alloc_v_subset();
  auto active_current = engine.alloc_v_subset();
  auto active_next    = engine.alloc_v_subset();
  auto parent         = engine.template alloc_v_state<plato::vid_t>();

  // init structs
  plato::vid_t invalid_parent = graph_info.max_v_i_ + 1;
  CHECK(invalid_parent != 0) << "vertex id overflow!";

  parent.fill(invalid_parent);
  parent[root] = root;

  visited.set_bit(root);
  active_current.set_bit(root);
  
  for (int epoch_i = 0; 0 != actives; ++epoch_i) {
    using pull_context_t = plato::template mepa_ag_context_t<plato::vid_t>;
    using pull_message_t = plato::template mepa_ag_message_t<plato::vid_t>;
    using push_context_t = plato::template mepa_bc_context_t<plato::vid_t>;
    using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;

    watch.mark("t1");
    active_next.clear();

    actives = engine.template foreach_edges<plato::vid_t, plato::vid_t> (
      [&](const push_context_t& context, vid_t v_i) {
        context.send(v_i);
      },
      [&](int /*p_i*/, plato::vid_t& msg) {
        plato::vid_t activated = 0;
        auto neighbours = out_edges.neighbours(msg);
        for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
          plato::vid_t dst = it->neighbour_;
          if (
            (parent[dst] == invalid_parent)
              &&
            (plato::cas(&parent[dst], invalid_parent, msg))
          ) {
            active_next.set_bit(dst);
            visited.set_bit(dst);
            ++activated;
          }
        }
        return activated;
      },
      [&](const pull_context_t& context, plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
        if (visited.get_bit(v_i)) { return ; }
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          plato::vid_t src = it->neighbour_;
          if (active_current.get_bit(src)) {
              context.send(pull_message_t { v_i, src});
              break;
          }
        }
      },
      [&](int, pull_message_t& msg) {
        if (plato::cas(&parent[msg.v_i_], invalid_parent, msg.message_)) {
          active_next.set_bit(msg.v_i_);
          visited.set_bit(msg.v_i_);
          return 1;
        }
        return 0;
      },
      active_current
    );

    auto active_view = plato::create_active_v_view(out_edges.partitioner()->self_v_view(), active_next);
    plato::vid_t __actives = active_view.template foreach<plato::vid_t>([&](plato::vid_t v_i) {
      visited.set_bit(v_i); return 1;
    });

    CHECK(__actives == actives) << "__actives: " << __actives << ", actives: " << actives;
    std::swap(active_current, active_next);

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "active_v[" << epoch_i << "] = " << actives << ", cost: " << watch.show("t1") / 1000.0 << "s";
    }
    if (stat.width_ < actives) { stat.width_ = actives; }
    stat.depth_++;
  }

  visited.sync();
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << " num of visit node is" <<visited.count();
  }
  {
    if (graph_info.is_directed_ == false) {
      if (graph_info.vertices_ != (graph_info.edges_ / 2 + 1)) {
        stat.is_tree_ = false;
      }
    }else {
      auto in_degrees = generate_dense_in_degrees_fg<plato::vid_t>(graph_info, in_edges, false);
      in_degrees.template foreach<plato::vid_t> ([&](plato::vid_t v_i, plato::vid_t *value) {
        if(*value >1) { stat.is_tree_ = false; }
        return 0;
      });
    }
    unsigned char tmp= stat.is_tree_;
    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, get_mpi_data_type<unsigned char>(), MPI_BAND, MPI_COMM_WORLD);
    stat.is_tree_ = static_cast<bool>(tmp);
  }
  return stat;
}

}}  // namespace algo, namespace plato

#endif

