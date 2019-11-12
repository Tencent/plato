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
#include <cfloat>
#include <unordered_map>
#include "gtest/gtest.h"
#include "plato/util/hyperloglog.hpp"
#include "plato/util/spinlock.hpp"

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <omp.h>
#include "mpi.h"
#include <queue>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"
#include "plato/algo/hyperanf/hyperanf.hpp"
#include "gtest_mpi_listener.hpp"

const std::string input = "data/graph/raw_graph_7_7.csv";     //input file, in csv format, without edge data.
const int32_t alpha = -1;                                     //alpha value used in sequence balance partition.
const bool is_directed = false;                                //is graph directed or not.
const bool part_by_in = true;                                 //partition by in-degree.
const int32_t step = 1;                                       //how many step's degree should be counted, -1 means infinity.
const int32_t bits = 6;                                       //hyperloglog bit width used for cardinality estimation.
const int32_t iterations = 20;                                //number of iterations.

template <typename STREAM_T>
void read_edge_list(STREAM_T& file, char sep,
                    std::function<void(int, int)> proc) {
  std::string line;
  while (getline(file, line)) {
    std::stringstream linestream(line);
    std::string value;

    int src;
    int dst;
    int count;
    for (count = 0; getline(linestream, value, sep); count++) {
      if (count == 0) {
        src = std::stoi(value);
      } else if (count == 1) {
        dst = std::stoi(value);
      }
    }
    if (count == 2) {
      proc(src, dst);
    } else {
      LOG(FATAL) << "format error: more than two vertices on an edge\n";
    }
  }  // end of outer while
}

void number_of_friends(std::map<int, std::vector<int> > & g, int n, int root,
  std::function<void(std::vector<int> &, int&)> proc) {
  std::vector<int> dist(n, -1);
  std::queue<int> q;
  q.push(root);
  dist[root] = 0;
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (auto &v: g[u]) {
      if (dist[v] != -1) continue;
      dist[v] = dist[u] + 1;
      q.push(v);
    }
  }

  int max_dis = 0;
  std::vector<int> friends(n, 0);
  max_dis = 0;
  for(int i = 0; i < n; ++i) {
    ++friends[dist[i]];
    if(max_dis < dist[i])
      max_dis = dist[i];
  }
  for(int i = 0; i < max_dis; ++i) {
    friends[i + 1] += friends[i];
  }
  proc(friends, max_dis);
}

static double get_avg_distance_by_src(const std::vector<int>& anfs, int max_dis) {
  double avg_distance = 0.0;

  for (size_t i = 1; i <= (size_t)max_dis; ++i) {
    avg_distance += i * (static_cast<double>(anfs[i]) - static_cast<double>(anfs[i - 1]));
  }

  avg_distance /= (static_cast<double>(anfs[max_dis]) - static_cast<double>(anfs[0]));
  return avg_distance;
}

void init_cluster_info() {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  cluster_info.sockets_      = 1;
  cluster_info.threads_ = (int)std::strtol(std::getenv("OMP_NUM_THREADS"), nullptr, 10);
  omp_set_dynamic(0);
  omp_set_num_threads(cluster_info.threads_);
  cluster_info.sockets_ = numa_num_configured_nodes();
  MPI_Comm_size(MPI_COMM_WORLD, &cluster_info.partitions_);
  MPI_Comm_rank(MPI_COMM_WORLD, &cluster_info.partition_id_);
}

TEST(Hyperanf, AvgDistance) {
  plato::stop_watch_t watch;
  using bcsr_spec_t = plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>;
  using dcsc_spec_t = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;
  init_cluster_info();

  plato::graph_info_t graph_info(is_directed);
  auto graph = plato::create_dualmode_seq_from_path<plato::empty_t>(&graph_info, input,
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>,
      alpha, part_by_in);

  plato::algo::hyperanf_opts_t opts;
  opts.iteration_ = iterations;

  watch.mark("t0");
  LOG(INFO) << "start" << std::endl;
  double avg_distance;
    switch (bits) {
    case 6:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 6>(graph.second, graph.first, graph_info, opts);
      break;

    case 7:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 7>(graph.second, graph.first, graph_info, opts);
      break;
    case 8:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 8>(graph.second, graph.first, graph_info, opts);
      break;
    case 9:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 9>(graph.second, graph.first, graph_info, opts);
      break;
    case 10:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 10>(graph.second, graph.first, graph_info, opts);
      break;
    case 11:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 11>(graph.second, graph.first, graph_info, opts);
      break;
    case 12:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 12>(graph.second, graph.first, graph_info, opts);
      break;
    case 13:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 13>(graph.second, graph.first, graph_info, opts);
      break;
    case 14:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 14>(graph.second, graph.first, graph_info, opts);
      break;
    case 15:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 15>(graph.second, graph.first, graph_info, opts);
      break;
    case 16:
      avg_distance = plato::algo::hyperanf<dcsc_spec_t, bcsr_spec_t, 16>(graph.second, graph.first, graph_info, opts);
      break;
    default:
      CHECK(false) << "unsupport hyperloglog bit width: " << bits
        << ", supported range is in [6, 16]";
  }

  std::map<int, std::vector<int>> g;
  int max_vid = 0;
  auto add_edge = [&](int src, int dst) {
    g[src].emplace_back(dst);
    g[dst].emplace_back(src);
    max_vid = std::max(max_vid, src);
    max_vid = std::max(max_vid, dst);
  };

  std::ifstream ifs(input, std::ifstream::in);
  read_edge_list(ifs, ',', add_edge);
  ifs.close();

  LOG(INFO) << "total vertex: " << max_vid + 1 << std::endl;

  double real_avg_distance = 0;
  for(int root = 0; root <= max_vid; ++root)
  {
    double person_avg_dis = 0;
    auto proc = [&](std::vector<int> & friends, int max_dis) {
      person_avg_dis = get_avg_distance_by_src(friends, max_dis);
    };
    number_of_friends(g, max_vid + 1, root, proc);
    real_avg_distance += person_avg_dis;
  }
  real_avg_distance /= (max_vid + 1);

  ASSERT_NEAR(real_avg_distance, avg_distance, 0.05);
  LOG(INFO) << "avg_distance:"  << avg_distance << std::endl;
  LOG(INFO) << "real_avg_distance:"  << real_avg_distance << std::endl;
}

int main(int argc, char** argv) {
  // Filter out Google Test arguments
  ::testing::InitGoogleTest(&argc, argv);

  // Initialize MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  // set OpenMP if not set
  if (getenv("OMP_NUM_THREADS") == nullptr) {
    setenv("OMP_NUM_THREADS", "2", 1);
  }

  // Add object that will finalize MPI on exit; Google Test owns this pointer
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

  // Get the event listener list.
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();

  // Remove default listener
  delete listeners.Release(listeners.default_result_printer());

  // Adds MPI listener; Google Test owns this pointer
  listeners.Append(new MPIMinimalistPrinter);

  // Run tests, then clean up and exit
  return RUN_ALL_TESTS();
}
