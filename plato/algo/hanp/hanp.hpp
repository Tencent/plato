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

#ifndef __PLATO_ALGO_hanp_HPP__
#define __PLATO_ALGO_hanp_HPP__

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include "glog/logging.h"
#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/util/spinlock.hpp"
#include "plato/graph/graph.hpp"

namespace plato { namespace algo {

struct hanp_opts_t {
  uint32_t iteration_ = 20;   // number of iterations
  double preference = 1.0;
  double hop_att = 0.1;
  double dis = 1e-6;
};

/*
 * run hanp on a graph with incoming edges
 * NOTICE: Currently HANP only works on sequence_by_destination partition
 * strategy.
 *
 * \tparam GRAPH  graph type, with incoming edges
 *
 * \param graph       the graph
 * \param graph_info  base graph-info
 * \param opts        hanp options
 *
 * \return
 *      each vertex's label value in dense representation
 **/
template <typename GRAPH>
dense_state_t<plato::vid_t, typename GRAPH::partition_t> hanp (
  GRAPH& graph,
  const graph_info_t& graph_info,
  const hanp_opts_t& opts = hanp_opts_t()) {

  using vid_t = plato::vid_t;
  using label_state_t  = plato::dense_state_t<vid_t, typename GRAPH::partition_t>;
  using hop_score_t  = plato::dense_state_t<float, typename GRAPH::partition_t>;
  using adj_unit_list_spec_t = typename GRAPH::adj_unit_list_spec_t;

  plato::allgather_state_opts_t allgather_opt;
  allgather_opt.threads_         = -1;
  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  /* initialize random generators for each thread*/
  std::vector<std::mt19937> rand_engines;
  for (int i = 0; i < cluster_info.threads_; i ++) {
    rand_engines.emplace_back(std::time(nullptr) + i);
  }

  auto rand_gen = [&](int tid) {
    return rand_engines[tid]();
  };

  /* init label */
  label_state_t curr_label = label_state_t(graph_info.max_v_i_, graph.partitioner());
  label_state_t prev_label = label_state_t(graph_info.max_v_i_, graph.partitioner());
  hop_score_t prev_att_score = hop_score_t(graph_info.max_v_i_, graph.partitioner());
  hop_score_t curr_att_score = hop_score_t(graph_info.max_v_i_, graph.partitioner());

  curr_label.template foreach<int>(
    [&](vid_t vtx, vid_t* pval) {
      *pval = vtx;
      return 1;
    }
  );

  prev_label.template foreach<int>(
    [&](vid_t vtx, vid_t* pval) {
      *pval = vtx;
      return 1;
    }
  );

  curr_att_score.template foreach<float>(
    [&](vid_t vtx, float* pval) {
      *pval = 1.0;
      return 1.0;
    }
  );

  prev_att_score.template foreach<float>(
    [&](vid_t vtx, float* pval) {
      *pval = 1.0;
      return 1.0;
    }
  );

  auto update_label_tid =
    [&](vid_t vtx, const adj_unit_list_spec_t& adjs, int tid) {
      std::unordered_map<vid_t, std::pair<float, float>> label_map;
      for (auto it = adjs.begin_; it != adjs.end_; it++) {
        if(prev_att_score[it->neighbour_] < 0) continue;
        vid_t nbr_label = prev_label[it->neighbour_];
        float nbr_score = prev_att_score[it->neighbour_] * opts.preference * it->edata_;
        auto search = label_map.find(nbr_label);
        if (search == label_map.end()) {
          label_map[nbr_label] = std::make_pair(nbr_score, prev_att_score[it->neighbour_]);
        } else {
          (search -> second).first += nbr_score;
          (search -> second).second = std::max((search -> second).second, prev_att_score[it->neighbour_]);
        }
      } /* end of for adj */

      std::vector<vid_t> max_labels;
      bool first = true;
      float max_score;
      for(auto kv : label_map) {
        if(first) {
          max_labels.push_back(kv.first);
          max_score = kv.second.first;
          first = false;
          continue;
        }

        if(kv.second.first > max_score) {
          max_labels.clear();
          max_labels.push_back(kv.first);
          max_score = kv.second.first;
        }
        else if(std::abs(kv.second.first - max_score) < opts.dis)
        {
          max_labels.push_back(kv.first);
        }
      }

      if (max_labels.size() > 0) {
        vid_t s = rand_gen(tid);
        curr_label[vtx] = max_labels[s % max_labels.size()];
      }
      if(prev_label[vtx] == curr_label[vtx]) {
        curr_att_score[vtx] = label_map[curr_label[vtx]].second;
      }
      else {
        curr_att_score[vtx] = label_map[curr_label[vtx]].second - opts.hop_att;
      }
      return true;
    };// end of update label tid

  /* iterate */
  for (uint32_t iter = 0; iter < opts.iteration_; iter ++) {
    watch.mark("t1");
    /* gather prev label to local */

    plato::allgather_state<vid_t>(prev_label, allgather_opt);
    plato::allgather_state<float>(prev_att_score, allgather_opt);

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "[epoch-" << iter << "] all gather stage done.";
    }

    graph.reset_traversal();
    /* update master's label */
    #pragma omp parallel num_threads(cluster_info.threads_)
    {
      int tid = omp_get_thread_num();
      size_t chunk_size = 16;

      auto update_label = [&](vid_t vtx, const adj_unit_list_spec_t& adjs) {
        return update_label_tid(vtx, adjs, tid);
      };

      while(graph.next_chunk(update_label, &chunk_size)) {
        ;
      }
    }

    std::swap(curr_label, prev_label);
    std::swap(curr_att_score, prev_att_score);

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "[epoch-" << iter << "], cost: " << watch.show("t1") / 1000.0 << "s";
    }

  }
  return curr_label;
}

}}  // namespace algo, namespace hanp

#endif

