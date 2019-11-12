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

#include <array>
#include <random>
#include <algorithm>
#include <functional>

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"

DEFINE_string(input,       "",    "input file, in csv format, without edge data");
DEFINE_bool(is_directed,   false,   "is graph directed or not");
DEFINE_bool(part_by_in,    true,  "partition by in-degree");
DEFINE_int32(alpha,        -1,    "alpha value used in sequence balance partition");
DEFINE_uint64(iterations,  10,    "number of iterations");

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

template <typename... Args>
void INFO_0(const char* f, Args&&... args) {
  if (0 == plato::cluster_info_t::get_instance().partition_id_) {
    int size = snprintf(nullptr, 0, f, args...);
    std::string res;
    res.resize(size);
    snprintf(&res[0], size + 1, f, args...);
    LOG(INFO) << res;
  }
}

int main(int argc, char** argv) {
  init(argc, argv);
  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.initialize(&argc, &argv);

  plato::stop_watch_t watch;

  watch.mark("t1");
  plato::graph_info_t graph_info(FLAGS_is_directed);
  auto pdcsc = plato::create_dcsc_seqs_from_path<plato::empty_t>(
    &graph_info, FLAGS_input, plato::edge_format_t::CSV,
    plato::dummy_decoder<plato::empty_t>, FLAGS_alpha, FLAGS_part_by_in
  );
  INFO_0("load cache cost: %lfs", watch.show("t1") / 1000.0);

  using graph_t              = std::remove_reference<decltype(*pdcsc)>::type;
  using feature_t            = std::array<float, 100>;
  using feature_state_t      = plato::dense_state_t<feature_t, typename graph_t::partition_t>;
  using context_spec_t       = plato::mepa_ag_context_t<feature_t>;
  using message_spec_t       = plato::mepa_ag_message_t<feature_t>;
  using adj_unit_list_spec_t = typename graph_t::adj_unit_list_spec_t;

  plato::dualmode_engine_t<graph_t, nullptr_t> engine (pdcsc, nullptr, graph_info);

  feature_state_t curt_rank = engine.template alloc_v_state<feature_t>();
  feature_state_t next_rank = engine.template alloc_v_state<feature_t>();

  curt_rank.template foreach<int> (
    [&](plato::vid_t v_i, feature_t* pval) {
      static thread_local std::mt19937 urng(std::random_device{}());
      std::uniform_real_distribution<float> dist(0, 1.0);
      pval->fill(dist(urng));
      return 0;
    }
  );

  watch.mark("t1");
  auto idegrees = plato::generate_dense_in_degrees_fg<uint32_t>(graph_info, *pdcsc, false);
  INFO_0("generate_dense_in_degrees cost: %lfs", watch.show("t1") / 1000.0);

  feature_t zeros;
  zeros.fill(0.0);

  for (uint32_t epoch_i = 0; epoch_i < FLAGS_iterations; ++epoch_i) {
    watch.mark("t1");

    next_rank.fill(zeros);
    engine.template foreach_edges<feature_t, int> (
      [&](const context_spec_t& context, plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
        feature_t agg = curt_rank[v_i];
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          auto& neisf = curt_rank[it->neighbour_];
          for (size_t i = 0; i < agg.size(); ++i) {
            agg[i] = agg[i] + neisf[i];
          }
        }
        context.send(message_spec_t { v_i, agg });
      },
      [&](int, message_spec_t& msg) {
        auto& hiddens = next_rank[msg.v_i_];
        for (size_t i = 0; i < hiddens.size(); ++i) {
          hiddens[i] = hiddens[i] + msg.message_[i];
        }
        return 0;
      }
    );

    next_rank.template foreach<int> (
      [&](plato::vid_t v_i, feature_t* pval) {
        for (size_t i = 0; i < pval->size(); ++i) {
          pval->at(i) /= idegrees[v_i];
        }
        return 0;
      }
    );

    INFO_0("epoch[%u], cost: %lf\n", epoch_i, watch.show("t1") / 1000.0);

    std::swap(curt_rank, next_rank);
  }

  return 0;
}

