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

#ifndef __PLATO_ALGO_hyperanf_HPP__
#define __PLATO_ALGO_hyperanf_HPP__

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <float.h>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/util/spinlock.hpp"
#include "plato/graph/graph.hpp"
#include "plato/util/hyperloglog.hpp"
#include "plato/engine/dualmode.hpp"

namespace plato { namespace algo {

struct hyperanf_opts_t {
  uint32_t iteration_ = 20;   // number of iterations
  uint32_t bits = 12;
};

template <uint32_t BitWidth>
struct hyperanf_msg_type_t {
  plato::vid_t vtx;
  plato::hyperloglog_t<BitWidth> hll_data;
  //plato::vid_t vtx2;
}__attribute__((__packed__));


/**
 * @brief
 * @param anfs
 * @return
 */
static double GetAvgDistance(const std::vector<double>& anfs) {
  size_t size = anfs.size();
  double avg_distance = 0.0;
  for (size_t i = 1; i < size; ++i) {
    avg_distance += i * (anfs[i] - anfs[i - 1]);
  }

  avg_distance /= (anfs.back() - anfs.front());
  return avg_distance;
}

/**
 * @brief
 * @tparam INCOMING
 * @tparam OUTGOING
 * @tparam BitWidth
 * @param in_edges
 * @param out_edges
 * @param graph_info
 * @param opts
 * @return
 */
template <typename INCOMING, typename OUTGOING, uint32_t BitWidth = 12>
double hyperanf (
  INCOMING& in_edges,
  OUTGOING& out_edges,
  const graph_info_t& graph_info,
  const hyperanf_opts_t& opts = hyperanf_opts_t()) {

  using vid_t = plato::vid_t;
  using hll = plato::hyperloglog_t<BitWidth>;
  plato::stop_watch_t watch;

  dualmode_engine_t<INCOMING, OUTGOING> engine (
    std::shared_ptr<INCOMING>(&in_edges,  [](INCOMING*) { }),
  std::shared_ptr<OUTGOING>(&out_edges, [](OUTGOING*) { }),
    graph_info);

  auto all_vertex     = engine.alloc_v_subset();
  auto active_current = engine.alloc_v_subset();
  auto active_next    = engine.alloc_v_subset();
  auto delta_active   = engine.alloc_v_subset();
  auto delta          = engine.template alloc_v_state<hll>();
  auto labels         = engine.template alloc_v_state<hll>();
  auto locks          = engine.template alloc_v_state<spinlock_noaligned_t>();

  std::vector<double> anfs;
  all_vertex.fill();
  active_current.fill();

  double init_anf = labels.template foreach<double>(
    [&](vid_t vtx, hll* hll_data) {
      hll tmphll;
      tmphll.init();
      *hll_data = tmphll;
      delta[vtx] = tmphll;
      hll_data->add(&vtx, sizeof(vid_t));
      new (&locks[vtx]) spinlock_noaligned_t;
      return hll_data->estimate();
    },
    &all_vertex);

  anfs.push_back(init_anf);
  for(uint32_t iter = 0; iter < opts.iteration_; ++iter) {
    using pull_context_t = plato::template mepa_ag_context_t<hyperanf_msg_type_t<BitWidth>>;
    using pull_message_t = plato::template mepa_ag_message_t<hyperanf_msg_type_t<BitWidth>>;
    using push_context_t = plato::template mepa_bc_context_t<hyperanf_msg_type_t<BitWidth>>;
    using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;
    active_next.clear();
    delta_active.clear();

    engine.template foreach_edges<hyperanf_msg_type_t<BitWidth>, int>(
      [&](const push_context_t& context, vid_t vtx) {
        context.send(hyperanf_msg_type_t<BitWidth>{ vtx, labels[vtx] });
      },
      [&](int /*p_i*/, hyperanf_msg_type_t<BitWidth>& msg) {
        auto neighbours = out_edges.neighbours(msg.vtx);
        for(auto it = neighbours.begin_; it != neighbours.end_; ++it) {
          vid_t dst = it->neighbour_;
          locks[dst].lock();
          if(delta[dst].merge(msg.hll_data) > 0) {
            delta_active.set_bit(dst);
          }
          locks[dst].unlock();
        }
        return 0;
      },
      [&](const pull_context_t& context, plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
        hll agg_nei;
        agg_nei.init();
        for(auto it = adjs.begin_; it != adjs.end_; ++it) {
          vid_t src = it->neighbour_;
          agg_nei.merge(labels[src]);
        }
        context.send(pull_message_t { v_i, hyperanf_msg_type_t<BitWidth>{ v_i, agg_nei }});
      },
      [&](int, pull_message_t& msg) {
        locks[msg.v_i_].lock();
        if(delta[msg.v_i_].merge(msg.message_.hll_data) > 0) {
          delta_active.set_bit(msg.v_i_);
        }
        locks[msg.v_i_].unlock();
        return 0;
      },
      active_current);

    labels.template foreach<int> (
      [&](vid_t vtx, hll* hll_data) {
        if(hll_data->merge(delta[vtx]) > 0) {
          active_next.set_bit(vtx);
        }
        return 0;
      },
      &delta_active);

    double anf = labels.template foreach<double> (
      [&](vid_t vtx, hll* hll_data) {
        hll tmphll;
        tmphll.init();
        delta[vtx] = tmphll;
        return hll_data->estimate();
      },
      &all_vertex);

    if(anf - anfs.back() < DBL_EPSILON) {
      break;
    }

    anfs.push_back(anf);
    std::swap(active_current, active_next);
  }

  double avg_distance = GetAvgDistance(anfs);
  return avg_distance;
}

}}

#endif

