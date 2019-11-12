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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <omp.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <queue>
#include <vector>

#include "gtest_mpi_listener.hpp"
#include "connected_component.hpp"

DEFINE_string(input, "data/graph/v100_e2150_ua_c3.csv", "data path");
DEFINE_int32(alpha, -1,     "alpha value used in sequence balance partition");
DEFINE_bool(is_directed,  false,    "is graph directed or not");
DEFINE_bool(part_by_in,  false,    "partition by in-degree");

using namespace plato;

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

void bfs(std::map<int, std::vector<int> > & g, int n, std::vector<int> & label) {
  for (int i = 0; i < n; ++i) {
    if (label[i] == -1) {
      label[i] = i;
      std::queue<int> q;
      q.push(i);
      while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (auto &v : g[u]) {
          if (label[v] != -1) continue;
          label[v] = i;
          q.push(v);
        }
      }
    }
  }
}

TEST(connected_component_t, calc_component) {
  //first calc in each single process
  // read
  std::map<int, std::vector<int>> g;
  int max_vid = 0;
  auto add_edge = [&](int src, int dst) {
    g[src].emplace_back(dst);
    g[dst].emplace_back(src);
    max_vid = std::max(max_vid, src);
    max_vid = std::max(max_vid, dst);
  };

  std::ifstream ifs(FLAGS_input, std::ifstream::in);
  read_edge_list(ifs, ',', add_edge);
  ifs.close();

  LOG(INFO) << "total vertex: " << max_vid + 1 << std::endl;
  std::vector<int> label(max_vid + 1, -1);
  bfs(g, max_vid + 1, label);

  //second calc in distribute
  init_cluster_info();
  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::graph_info_t graph_info(FLAGS_is_directed);
  auto graph = plato::create_dualmode_seq_from_path<plato::empty_t>(
    &graph_info, FLAGS_input,
    plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>,
    FLAGS_alpha, FLAGS_part_by_in);

  using bcsr_spec_t = plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>;
  using dcsc_spec_t = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;

  dualmode_engine_t<dcsc_spec_t, bcsr_spec_t> engine (
    std::shared_ptr<dcsc_spec_t>(&graph.second,  [](dcsc_spec_t*) { }),
    std::shared_ptr<bcsr_spec_t>(&graph.first, [](bcsr_spec_t*) { }),
    graph_info);

  plato::algo::connected_component_t<dcsc_spec_t, bcsr_spec_t> cc(&engine, graph_info);
  cc.compute();

  //LOG(INFO) << "[summary]: " << cc.get_summary() << std::endl;

  auto local_label = cc.get_labels();
  for (int i = 0; i <= max_vid; i++) {
    int pid = engine.out_edges()->partitioner()->get_partition_id((plato::vid_t)i);
    if (pid == cluster_info.partition_id_) {
      //LOG(INFO) << "process: " << pid << " vid: " << i << " label: " << label[i] << std::endl;
      ASSERT_EQ((*local_label)[i], label[i]);
    }
  }

  LOG(INFO) << "test pass!" << std::endl;
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
