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
#include "betweenness.hpp"

DEFINE_string(input, "data/graph/v100_e2150_ua_c3.csv", "data path");
DEFINE_int32(alpha, -1,     "alpha value used in sequence balance partition");
DEFINE_bool(is_directed,  false,    "is graph directed or not");
DEFINE_bool(part_by_in,  false,    "partition by in-degree");
DEFINE_int32(chosen, -1, "chosen vertex");
DEFINE_int32(max_iteration, 0, "");
DEFINE_double(constant, 2, "");

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

//serial algorithm
void calc(std::map<int, std::vector<int> > & g, vid_t n, int m,
    std::unordered_set<vid_t>* samples, std::vector<double>& betweenness) {
  for (auto &root : *samples) {
    //first calc num_paths and get bfs tree
    LOG(INFO) << "root: " << root << std::endl;
    std::vector<int> num_paths(n, 0);
    std::vector<bool> visited(n, 0);
    std::vector<std::vector<int> > levels;
    std::vector<double> dependencies(n, 0);
    num_paths[root] = 1;
    visited[root] = true;
    std::queue<int> q;
    q.push(root);
    while (true) {
      std::vector<int> cur;
      while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (auto &v : g[u]) {
          if (!visited[v]) {
            if (num_paths[v] == 0) cur.push_back(v);
            num_paths[v] += num_paths[u];
          }
        }
      }
      LOG(INFO) << "level size: " << cur.size() << std::endl;
      if (cur.size() == 0) break;
      levels.push_back(cur);
      for (auto &v : cur) {
        visited[v] = true;
        q.push(v);
      }
    }
    //second get betweenness of each vertex
    std::fill(visited.begin(), visited.end(), false);
    while (levels.size() > 0) {
      const auto &cur = levels.back();
      for (const auto &u : cur) {
        visited[u] = true;
      }
      for (const auto &u : cur) {
        for (auto &v : g[u]) {
          if (!visited[v]) {
            dependencies[v] += ((dependencies[u] + 1.0) * (double)num_paths[v] / num_paths[u]);
          }
        }
      }
      levels.pop_back();
    }
    for (vid_t i = 0; i < n; ++i) {
      if (i != root) betweenness[i] += dependencies[i];
    }
  }
  for (vid_t i = 0; i < n; ++i) {
    betweenness[i] = betweenness[i] * m / samples->size();
  }
}

TEST(betweenness, bader) {
  //first calc in distribute
  init_cluster_info();
  plato::graph_info_t graph_info(FLAGS_is_directed);
  auto graph = plato::create_dualmode_seq_from_path<plato::empty_t>(&graph_info, FLAGS_input,
    plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>,
    FLAGS_alpha, FLAGS_part_by_in);

  using bcsr_spec_t = plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>;
  using dcsc_spec_t = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;

  dualmode_engine_t<dcsc_spec_t, bcsr_spec_t> engine (
    std::shared_ptr<dcsc_spec_t>(&graph.second,  [](dcsc_spec_t*) { }),
    std::shared_ptr<bcsr_spec_t>(&graph.first, [](bcsr_spec_t*) { }),
    graph_info);

  plato::algo::bader_opts_t opts;
  opts.max_iteration_ = FLAGS_max_iteration;
  opts.chosen_ = FLAGS_chosen;
  opts.constant_ =  FLAGS_constant;
  plato::algo::bader_betweenness_t<dcsc_spec_t, bcsr_spec_t, double> bader(&engine, graph_info, opts);
  bader.compute();

  //second calc in each single process
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
  LOG(INFO) << "sample size: " << bader.get_samples()->size() << std::endl;
  std::vector<double> betweenness(max_vid + 1, 0.0);
  //we already know the samples through bader algorithm, so just calc it
  calc(g, max_vid + 1, bader.get_componnent_vertices(), bader.get_samples(), betweenness); 
  
  //use brute force to check
  for (int i = 0; i <= max_vid; ++i) {
    double val1 = betweenness[i];
    double val2 = bader.get_betweenness_of(i);
    LOG(INFO) << "vid: " << i << " val1: " << val1 << " val2: " << val2 << std::endl;
    ASSERT_NEAR(val1, val2, 1e-7);
  }
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
