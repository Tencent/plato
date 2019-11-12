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
#include <unordered_set>

#include "gtest_mpi_listener.hpp"
#include "closeness.hpp"

DEFINE_string(input, "data/graph/v100_e2150_ua_c3.csv", "data path");
DEFINE_int32(alpha, -1,     "alpha value used in sequence balance partition");
DEFINE_bool(is_directed,  false,    "is graph directed or not");
DEFINE_bool(part_by_in,  false,    "partition by in-degree");
DEFINE_int32(num_samples, 10, "number of nodes to test");

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

void bfs(std::map<int, std::vector<int> > & g, int n, int root,
    std::function<void(std::vector<int> &)> proc) {
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
  proc(dist);
}

TEST(closeness, bavelas) {
  //store edge in each single process
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

  //init graph
  init_cluster_info();
  auto& cluster_info = plato::cluster_info_t::get_instance();
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

  plato::algo::bavelas_closeness_t<dcsc_spec_t, bcsr_spec_t> bavelas(&engine, graph_info);

  //make samples
  std::vector<int> samples(FLAGS_num_samples);
  if (cluster_info.partition_id_ == 0) {
    std::unordered_set<int> unique_samples;
    while ((int)unique_samples.size() < FLAGS_num_samples) {
      int r = std::rand() % (max_vid + 1) ;
      unique_samples.insert(r);
    }
    std::copy(unique_samples.begin(), unique_samples.end(), samples.begin());
  }
  MPI_Bcast(&samples[0], FLAGS_num_samples, MPI_INT, 0, MPI_COMM_WORLD);

  // use brute force to check 
  for (auto &root: samples) {
    uint64_t sum1 = 0; 
    auto proc = [&](std::vector<int> & dist) {
      for (auto &val: dist) {
        if (val > 0) sum1 += val;
      } 
    };
    bfs(g, max_vid + 1, root, proc);
    if (sum1 == 0) continue;
    double closeness1 = 1.0 * max_vid / sum1;
    double closeness2 = bavelas.compute((vid_t)root);
    LOG(INFO) << "root: " << root << " cl1: " << closeness1 << " cl2: " << closeness2 << std::endl;
    ASSERT_NEAR(closeness1, closeness2, 1e-7);
  }
}

TEST(closeness, david) {
  //store edge in each single process
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

  //init graph
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

  plato::algo::david_opts_t opts;
  opts.num_samples_ = FLAGS_num_samples;
  plato::algo::david_closeness_t<dcsc_spec_t, bcsr_spec_t> david(&engine, graph_info, opts);
  
  david.compute();
  std::vector<double> cl(max_vid + 1, 0);
  std::vector<bool> visited(max_vid + 1, false);
  auto proc = [&](std::vector<int> & dist) {
    for (int i = 0; i <= max_vid; ++i) {
      if (dist[i] < 0) continue;
      cl[i] += dist[i];
      visited[i] = true;
    }
  };
  for (auto &root: *(david.get_samples())) {
    bfs(g, max_vid + 1, root, proc);
  }

  int major_cnt = david.get_major_component_vertices();
  for (int i = 0; i <= max_vid; ++i) {
    if (!visited[i]) continue;
    double val1 = 1.0 * FLAGS_num_samples * (major_cnt - 1) / major_cnt / cl[i];
    double val2 = david.get_closeness_of(i);
    LOG(INFO) << "root: " << i << " val1: " << val1 << " val2: " << val2 << std::endl;
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
