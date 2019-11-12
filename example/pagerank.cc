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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/hdfs.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/base.hpp"
#include "plato/graph/state.hpp"
#include "plato/graph/structure.hpp"
#include "plato/graph/message_passing.hpp"

DEFINE_string(input,       "",      "input file, in csv format, without edge data");
DEFINE_string(output,      "",      "output directory");
DEFINE_bool(is_directed,   false,   "is graph directed or not");
DEFINE_bool(part_by_in,    false,   "partition by in-degree");
DEFINE_int32(alpha,        -1,      "alpha value used in sequence balance partition");
DEFINE_uint64(iterations,  100,     "number of iterations");
DEFINE_double(damping,     0.85,    "the damping factor");
DEFINE_double(eps,         0.001,   "the calculation will be consider \
                                      as complete if the difference of PageRank values between iterations \
                                      change less than this value for every node");

bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

DEFINE_validator(input,  &string_not_empty);
DEFINE_validator(output, &string_not_empty);

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

int main(int argc, char** argv) {
  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  watch.mark("t0");

  // init graph
  plato::graph_info_t graph_info(FLAGS_is_directed);
  auto pdcsc = plato::create_dcsc_seqs_from_path<plato::empty_t>(
    &graph_info, FLAGS_input, plato::edge_format_t::CSV,
    plato::dummy_decoder<plato::empty_t>, FLAGS_alpha, FLAGS_part_by_in
  );

  using graph_spec_t         = std::remove_reference<decltype(*pdcsc)>::type;
  using partition_t          = graph_spec_t::partition_t;
  using adj_unit_list_spec_t = graph_spec_t::adj_unit_list_spec_t;
  using rank_state_t         = plato::dense_state_t<double, partition_t>;

  // init state
  std::shared_ptr<rank_state_t> curt_rank(new rank_state_t(graph_info.max_v_i_, pdcsc->partitioner()));
  std::shared_ptr<rank_state_t> next_rank(new rank_state_t(graph_info.max_v_i_, pdcsc->partitioner()));

  watch.mark("t1");
  auto odegrees = plato::generate_dense_out_degrees_fg<uint32_t>(graph_info, *pdcsc, false);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "generate out-degrees from graph cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  watch.mark("t2"); // do computation

  double delta = curt_rank->foreach<double> (
    [&](plato::vid_t v_i, double* pval) {
      *pval = 1.0;
      if (odegrees[v_i] > 0) {
        *pval = *pval / odegrees[v_i];
      }
      return 1.0;
    }
  );

  using context_spec_t = plato::mepa_ag_context_t<double>;
  using message_spec_t = plato::mepa_ag_message_t<double>;

  for (uint32_t epoch_i = 0; epoch_i < FLAGS_iterations; ++epoch_i) {
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "delta: " << delta;
    }

    watch.mark("t1");
    next_rank->fill(0.0);

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "[epoch-" << epoch_i  << "] init-next cost: "
        << watch.show("t1") / 1000.0 << "s";
    }

    watch.mark("t1");
    plato::aggregate_message<double, int, graph_spec_t> (*pdcsc,
      [&](const context_spec_t& context, plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
        double rank_sum = 0.0;
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          rank_sum += (*curt_rank)[it->neighbour_];
        }
        context.send(message_spec_t { v_i, rank_sum });
      },
      [&](int /*p_i*/, message_spec_t& msg) {
        plato::write_add(&(*next_rank)[msg.v_i_], msg.message_);
        return 0;
      }
    );

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "[epoch-" << epoch_i  << "] message-passing cost: "
        << watch.show("t1") / 1000.0 << "s";
    }

    watch.mark("t1");
    if (FLAGS_iterations - 1 == epoch_i) {
      delta = next_rank->foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          *pval = 1.0 - FLAGS_damping + FLAGS_damping * (*pval);
          return 0;
        }
      );
    } else {
      delta = next_rank->foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          *pval = 1.0 - FLAGS_damping + FLAGS_damping * (*pval);
          if (odegrees[v_i] > 0) {
            *pval = *pval / odegrees[v_i];
            return fabs(*pval - (*curt_rank)[v_i]) * odegrees[v_i];
          }
          return fabs(*pval - (*curt_rank)[v_i]);
        }
      );

      if (FLAGS_eps > 0.0 && delta < FLAGS_eps) {
        epoch_i = FLAGS_iterations - 2;
      }
    }
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "[epoch-" << epoch_i  << "] foreach_vertex cost: "
        << watch.show("t1") / 1000.0 << "s";
    }
    std::swap(curt_rank, next_rank);
  }

  delta = curt_rank->foreach<double> (
    [&](plato::vid_t v_i, double* pval) {
      return *pval;
    }
  );

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "iteration done, cost: " << watch.show("t2") / 1000.0 << "s, rank-sum: " << delta;
    LOG(INFO) << "whole cost: " << watch.show("t0") / 1000.0 << "s";
  }

  watch.mark("t1");
  {  // save result to hdfs
    std::vector<std::unique_ptr<plato::hdfs_t::fstream>> fs_v(cluster_info.threads_);
    std::vector<std::unique_ptr<boost::iostreams::filtering_stream<boost::iostreams::output>>>
      fs_output_v(cluster_info.threads_);

    for (int i = 0; i < cluster_info.threads_; ++i) {
      fs_v[i].reset(new plato::hdfs_t::fstream(plato::hdfs_t::get_hdfs(FLAGS_output),
          (boost::format("%s/%04d_%04d.csv.gz") % FLAGS_output.c_str() % cluster_info.partition_id_ % i).str(), true));

      fs_output_v[i].reset(new boost::iostreams::filtering_stream<boost::iostreams::output>());
      fs_output_v[i]->push(boost::iostreams::gzip_compressor());
      fs_output_v[i]->push(*fs_v[i]);
    }

    curt_rank->foreach<int> (
      [&](plato::vid_t v_i, double* pval) {
        static thread_local auto& fs_output = fs_output_v[omp_get_thread_num()];
        *fs_output << v_i << "," << *pval << "\n";
        return 0;
      }
    );
  }
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "save result cost: " << watch.show("t1") / 1000.0 << "s";
  }

  plato::mem_status_t mstatus;
  plato::self_mem_usage(&mstatus);
  LOG(INFO) << "memory usage: " << (double)mstatus.vm_rss / 1024.0 << " MBytes";

  return 0;
}

