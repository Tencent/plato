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

#ifndef __PLATO_ALGO_CNC_BAVELAS_HPP__
#define __PLATO_ALGO_CNC_BAVELAS_HPP__

#include <stdint.h>

#include "distance.hpp"

// the standard closeness centrality computation
//
// Closeness was defined by Bavelas (1950) as the reciprocal of the farness

namespace plato { namespace algo {

template <typename INCOMING, typename OUTGOING>
class bavelas_closeness_t {
public:
  explicit bavelas_closeness_t(dualmode_engine_t<INCOMING, OUTGOING> * engine,
      const graph_info_t& graph_info);

  double compute(vid_t root);

private:
  dualmode_engine_t<INCOMING, OUTGOING> * engine_;
  graph_info_t graph_info_;
};

template <typename INCOMING, typename OUTGOING>
bavelas_closeness_t<INCOMING, OUTGOING>::bavelas_closeness_t(
    dualmode_engine_t<INCOMING, OUTGOING> * engine, 
    const graph_info_t& graph_info) : engine_(engine), graph_info_(graph_info) {
}

template <typename INCOMING, typename OUTGOING>
double bavelas_closeness_t<INCOMING, OUTGOING>::compute(vid_t root) {
  uint64_t sum = 0;
  auto calc_func = [&](vid_t v_i, vid_t * val) {
    return *val;
  };

  sum = calc_distance<INCOMING, OUTGOING>(engine_, root, calc_func);
  if (sum == 0) return 0;
  double closeness = 1.0 * (graph_info_.vertices_ - 1) / sum;
  return closeness;
}

}  // namespace algo
}  // namespace plato
#endif 
