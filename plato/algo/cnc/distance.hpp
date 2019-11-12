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

#ifndef __PLATO_ALGO_CNC_DISTANCE_HPP__
#define __PLATO_ALGO_CNC_DISTANCE_HPP__

#include <algorithm>
#include <functional>
#include <utility>

#include "glog/logging.h"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"

namespace plato { namespace algo {
struct distance_msg_type_t {
  vid_t src_;
  vid_t dist_;
};

template <typename INCOMING, typename OUTGOING>
uint64_t calc_distance(
    dualmode_engine_t<INCOMING, OUTGOING> * engine, vid_t root,
    std::function<uint64_t(vid_t , vid_t *)> traversal) {

  LOG(INFO) << "calc distance of root: " << root << std::endl;
  auto dist = engine->template alloc_v_state<vid_t>();
  auto visited = engine->alloc_v_subset();
  auto active_current = engine->alloc_v_subset();
  auto active_next    = engine->alloc_v_subset();

  visited.clear();
  visited.set_bit(root);
  active_current.clear();
  active_current.set_bit(root);

  const vid_t INIT_VALUE = static_cast<vid_t>(0);
  dist.fill(INIT_VALUE);

  vid_t active_vertices = 1;

  using pull_context_t = plato::template mepa_ag_context_t<distance_msg_type_t>;
  using pull_message_t = plato::template mepa_ag_message_t<distance_msg_type_t>;
  using push_context_t = plato::template mepa_bc_context_t<distance_msg_type_t>;
  using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;

  for (int i_i = 0; active_vertices > 0; i_i++) {
    active_next.clear();
    active_vertices = engine->template foreach_edges<distance_msg_type_t, vid_t>(
      [&](const push_context_t& context, vid_t v_i) {
        context.send(distance_msg_type_t{v_i, dist[v_i]});
      },
      [&](int /*p_i*/, distance_msg_type_t & msg) {
        vid_t activated = 0;
        auto neighbours = engine->out_edges()->neighbours(msg.src_);
        for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
          vid_t dst = it->neighbour_;
          if (!visited.get_bit(dst) && cas(&dist[dst], INIT_VALUE, msg.dist_ + 1)) {
            active_next.set_bit(dst);
            ++activated;
          }
        }
        return activated;
      },
      [&](const pull_context_t& context, vid_t v_i, const adj_unit_list_spec_t& adjs) {
        if (visited.get_bit(v_i)) return;
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          vid_t src = it->neighbour_;
          if (active_current.get_bit(src)) {
            context.send(pull_message_t { v_i, distance_msg_type_t{ v_i, dist[src]} });
            break;
          }
        }
      },
      [&](int, pull_message_t& msg) {
        if (cas(&dist[msg.v_i_], INIT_VALUE, msg.message_.dist_ + 1)) {
          active_next.set_bit(msg.v_i_);
          return 1;
        }
        return 0;
      },
      active_current
    );
    auto active_view = plato::create_active_v_view(engine->out_edges()->partitioner()->self_v_view(), active_next);
    active_view.template foreach<vid_t>([&](vid_t v_i) {
      visited.set_bit(v_i); return 1;
    });
    std::swap(active_current, active_next);
  }

  uint64_t sum = dist.template foreach<uint64_t>(traversal);
  return sum;
}

}  // namespace algo 
}  // namespace plato
#endif 
