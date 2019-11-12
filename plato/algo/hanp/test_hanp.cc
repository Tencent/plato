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
#include <ctime>

#include "gtest_mpi_listener.hpp"
#include "plato/algo/hanp/hanp.hpp"
#include "gtest/gtest.h"
#include "plato/util/spinlock.hpp"

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include "mpi.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
using namespace plato;

const std::string input = "data/graph/raw_graph_10_9_weighted.csv";     //input file, in csv format, without edge data.
const int32_t alpha = -1;                                               //alpha value used in sequence balance partition.
const bool is_directed = false;                                         //is graph directed or not.
const bool part_by_in = true;                                           //partition by in-degree.
const int32_t iterations = 20;                                          //number of iterations.
const float preference = 1.0;                                           //is any arbitrary comparable characteristic for any node.
const float hop_att = 0.05;                                             //a new attenuated score.
const float dis = 1e-6;

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
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

template <typename STREAM_T>
void read_edge_list(STREAM_T& file, char sep,
                    std::function<void(int, int, float)> proc) {
  std::string line;
  while (getline(file, line)) {
    std::stringstream linestream(line);
    std::string value;
    int src;
    int dst;
    float e_data;
    int count;
    for (count = 0; getline(linestream, value, sep); count++) {
      if (count == 0) {
        src = std::stoi(value);
      } else if (count == 1) {
        dst = std::stoi(value);
      } else if (count == 2) {
        e_data = std::stof(value);
      }
    }
    if (count == 3) {
      proc(src, dst, e_data);
    } else {
      LOG(FATAL) << "format error: more than two vertices on an edge\n";
    }
  }  // end of outer while
}

TEST(HANP, LABELS)
{
  srand((unsigned)time(NULL));
  init_cluster_info();
  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::graph_info_t graph_info(is_directed);
  auto pdcsc = plato::create_dcsc_seqd_from_path<float>(
    &graph_info, input, plato::edge_format_t::CSV,
    plato::float_decoder, alpha, part_by_in
  );

  using graph_spec_t = std::remove_reference<decltype(*pdcsc)>::type;

  plato::algo::hanp_opts_t opts;
  opts.iteration_ = iterations;
  opts.preference = preference;
  opts.hop_att    = hop_att;
  opts.dis        = 1e-6;

  auto labels = plato::algo::hanp<graph_spec_t>(*pdcsc, graph_info, opts);
  
  labels.template foreach<int> (
    [&] (plato::vid_t v_i, plato::vid_t* label) {
      LOG(INFO) << v_i << "parallel" <<"--------->" << *label;
      return 0;
    });

  std::map<int, std::vector<std::pair<vid_t, float>>> g;
  int max_vid = 0;
  auto add_edge = [&](int src, int dst, float e_data) { 
    g[src].emplace_back(std::make_pair(dst, e_data));
    g[dst].emplace_back(std::make_pair(src, e_data)); 
    max_vid = std::max(max_vid, src);
    max_vid = std::max(max_vid, dst);
  };
  
  std::ifstream ifs(input, std::ifstream::in);
  read_edge_list(ifs, ',', add_edge);
  ifs.close();
  LOG(INFO) << "total vertex: " << max_vid + 1 << std::endl;

  std::vector<plato::vid_t> prev_labels(max_vid + 1, 0);
  std::vector<plato::vid_t> curr_labels(max_vid + 1, 0);
  std::vector<float> prev_att_score(max_vid + 1, 1.0);
  std::vector<float> curr_att_score(max_vid + 1, 1.0);
  for(int i = 0; i <= max_vid; ++i) {
    prev_labels[i] = i;
    curr_labels[i] = i;
  }
  
  auto compute = [&] (vid_t v_i) {
    std::unordered_map<vid_t, std::pair<float, float>> label_map;
    for(auto it : g[v_i]) {
      vid_t src = it.first;
      if(prev_att_score[src] < 0) continue;
      float e_data = it.second;
      vid_t nr_label = prev_labels[src];
      float nr_score = prev_att_score[src] * e_data;
      auto search = label_map.find(nr_label);
      if(search == label_map.end()) {
        label_map[nr_label] = std::make_pair(nr_score, prev_att_score[src]);
      } else {
        (search->second).first += nr_score;
        (search->second).second = std::max(prev_att_score[src], (search->second).second);
      }
    }

    std::vector<vid_t> max_labels;
    float max_score;
    bool first = true;
    for(auto kv : label_map) {
      if(first) {
        first = false;
        max_labels.push_back(kv.first);
        max_score = kv.second.first;
      }

      if(max_score < kv.second.first) {
        max_score = kv.second.first;
        max_labels.clear();
        max_labels.push_back(kv.first);
      } else if (std::abs(max_score - kv.second.first) < dis) {
        max_labels.push_back(kv.first);
      }
    }

    if (max_labels.size() > 0) {
      vid_t s = rand() % max_labels.size();
      curr_labels[v_i] = max_labels[s];
    }
    if(prev_labels[v_i] == curr_labels[v_i]) {
      curr_att_score[v_i] = label_map[curr_labels[v_i]].second;
    }
    else {
      curr_att_score[v_i] = label_map[curr_labels[v_i]].second - hop_att;
    }
  };
  
  for(int i = 0; i < iterations; ++i) {
    for(int j = 0; j <= max_vid; ++j) {
      compute(j);
    }
    std::swap(curr_labels, prev_labels);
    std::swap(curr_att_score, prev_att_score);
  }

  for(int i = 0; i <= max_vid; ++i) {
    LOG(INFO) << i << "serial" << "---------->" << curr_labels[i];
  }
  
  labels.template foreach<int> (
    [&] (plato::vid_t v_i, plato::vid_t* label) {
      EXPECT_EQ(curr_labels[v_i], *label);
      return 0;
    });
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
