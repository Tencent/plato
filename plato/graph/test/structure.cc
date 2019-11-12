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

#include "plato/graph/structure.hpp"

#include <unistd.h>

#include <mutex>
#include <vector>
#include <atomic>
#include <utility>
#include <fstream>
#include <algorithm>

#include "mpi.h"
#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gflags/gflags.h"
#include "gtest_mpi_listener.hpp"

#include "plato/graph/partition/hash.hpp"
#include "plato/graph/structure/edge_cache.hpp"

using namespace plato;

namespace plato {

inline bool operator==(const plato::edge_unit_t<plato::empty_t>& lhs, const plato::edge_unit_t<plato::empty_t>& rhs) {
  return (lhs.src_ == rhs.src_) && (lhs.dst_ == rhs.dst_);
}

template <typename T>
inline bool operator==(const plato::edge_unit_t<T>& lhs, const plato::edge_unit_t<T>& rhs) {
  return (lhs.src_ == rhs.src_) && (lhs.dst_ == rhs.dst_) && (lhs.edata_ == rhs.edata_);
}

}

void init_cluster_info(void) {
  auto& cluster_info = cluster_info_t::get_instance();

  cluster_info.partitions_   = 1;
  cluster_info.partition_id_ = 0;
  cluster_info.threads_      = 3;
  cluster_info.sockets_      = 1;
}

void load_plain_edges_weightless(const std::string& path, std::vector<edge_unit_t<empty_t>>* pedges) {
  std::fstream fs(path, std::fstream::in);
  std::mutex mtx;

  auto callback = [&](edge_unit_t<empty_t>* pInput, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      std::unique_lock<std::mutex> lck(mtx);
      pedges->emplace_back(pInput[i]);
    }
    return true;
  };

  csv_parser<std::fstream, empty_t, plato::vid_t>(fs, callback, dummy_decoder<empty_t>);
}

void load_plain_edges_float_undirected(const std::string& path, std::vector<edge_unit_t<float>>* pedges) {
  std::fstream fs(path, std::fstream::in);
  std::mutex mtx;

  auto callback = [&](edge_unit_t<float>* pInput, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      std::unique_lock<std::mutex> lck(mtx);
      pedges->emplace_back(pInput[i]);
    }
    return true;
  };

  csv_parser<std::fstream, float, plato::vid_t>(fs, callback, float_decoder);
}

TEST(GraphOps, LoadEdgesWeightedUndirectedToCache) {
  init_cluster_info();

  graph_info_t ginfo;
  auto cache = load_edges_cache<float>(&ginfo, "data/graph/graph_10_9.csv", edge_format_t::CSV, float_decoder);
  ASSERT_NE(nullptr, cache);

  ASSERT_EQ(10, ginfo.vertices_);
  ASSERT_EQ(9,  ginfo.edges_);

  std::vector<edge_unit_t<float>> edges;
  load_plain_edges_float_undirected("data/graph/graph_10_9.csv", &edges);
  ASSERT_EQ(cache->size(), edges.size());

  std::atomic<size_t> edge_count(0);
  auto traversal = [&](size_t, edge_unit_t<float>* edge) {
    ++edge_count;
    [&](void) { ASSERT_TRUE(edges.end() != std::find(edges.begin(), edges.end(), *edge)); } ();
    return true;
  };

  size_t chunk_size = 3;
  cache->reset_traversal();
  while (cache->next_chunk(traversal, &chunk_size)) { }
  ASSERT_EQ(edge_count, edges.size());

  // do traverse parallel

  edge_count.store(0);
  cache->reset_traversal();

  #pragma omp parallel
  {
    chunk_size = 3;
    while (cache->next_chunk(traversal, &chunk_size)) { }
  }

  ASSERT_EQ(edge_count, edges.size());
}

TEST(GraphOps, CountOutDenseDegreeUndirected) {
  init_cluster_info();

  graph_info_t graph_info;
  graph_info.is_directed_ = false;

  auto cache = load_edges_cache<empty_t>(&graph_info, "data/graph/v100_e2150_ua_c3.csv", edge_format_t::CSV, dummy_decoder<empty_t>);

  std::vector<uint32_t> degrees = generate_dense_out_degrees<uint32_t>(graph_info, *cache);

  ASSERT_EQ(degrees.size(), 100);

  ASSERT_EQ(degrees[0],  19);
  ASSERT_EQ(degrees[99], 59);
  ASSERT_EQ(degrees[28], 19);
}

TEST(GraphOps, CountOutDenseDegreeDirected) {
  init_cluster_info();

  graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = load_edges_cache<empty_t>(&graph_info, "data/graph/v100_e4300_da_c3.csv", edge_format_t::CSV, dummy_decoder<empty_t>);

  std::vector<uint32_t> degrees = generate_dense_out_degrees<uint32_t>(graph_info, *cache);

  ASSERT_EQ(degrees.size(), 100);

  ASSERT_EQ(degrees[0],  19);
  ASSERT_EQ(degrees[99], 59);
  ASSERT_EQ(degrees[28], 19);
}

TEST(GraphOps, CountInDenseDegreeUndirected) {
  init_cluster_info();

  graph_info_t graph_info;
  graph_info.is_directed_ = false;

  auto cache = load_edges_cache<empty_t>(&graph_info, "data/graph/v100_e2150_ua_c3.csv", edge_format_t::CSV, dummy_decoder<empty_t>);

  std::vector<uint32_t> degrees = generate_dense_in_degrees<uint32_t>(graph_info, *cache);

  ASSERT_EQ(degrees.size(), 100);

  ASSERT_EQ(degrees[0],  19);
  ASSERT_EQ(degrees[99], 59);
  ASSERT_EQ(degrees[28], 19);
}

TEST(GraphOps, CountInDenseDegreeDirected) {
  init_cluster_info();

  graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = load_edges_cache<empty_t>(&graph_info, "data/graph/v100_e4300_da_c3.csv", edge_format_t::CSV, dummy_decoder<empty_t>);

  std::vector<uint32_t> degrees = generate_dense_in_degrees<uint32_t>(graph_info, *cache);

  ASSERT_EQ(degrees.size(), 100);

  ASSERT_EQ(degrees[0],  19);
  ASSERT_EQ(degrees[99], 59);
  ASSERT_EQ(degrees[28], 19);
}

TEST(GraphOps, CountOutDenseDegreeDirectedFromInGraph) {
  init_cluster_info();

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto pdcsc = plato::create_dcsc_seqs_from_path<empty_t>(&graph_info, "data/graph/v100_e4300_da_c3.csv",
    plato::edge_format_t::CSV, dummy_decoder<empty_t>);

  auto degrees = plato::generate_dense_out_degrees_fg<uint32_t>(graph_info, *pdcsc, false);

  ASSERT_EQ(degrees[0],  19);
  ASSERT_EQ(degrees[99], 59);
  ASSERT_EQ(degrees[28], 19);
}

TEST(GraphOps, CountOutDenseDegreeDirectedFromOutGraph) {
  init_cluster_info();

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto pbcsr = plato::create_bcsr_seqd_from_path<empty_t>(&graph_info, "data/graph/v100_e4300_da_c3.csv",
    plato::edge_format_t::CSV, dummy_decoder<empty_t>);

  auto degrees = plato::generate_dense_out_degrees_fg<uint32_t>(graph_info, *pbcsr, true);

  ASSERT_EQ(degrees[0],  19);
  ASSERT_EQ(degrees[99], 59);
  ASSERT_EQ(degrees[28], 19);
}

TEST(GraphOps, CountSparseOutDegreeDirected) {
  init_cluster_info();

  using part_spec_t = plato::hash_by_source_t<>;

  graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = load_edges_cache<empty_t>(&graph_info, "data/graph/v100_e4300_da_c3.csv", edge_format_t::CSV, dummy_decoder<empty_t>);

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  auto degrees = generate_sparse_out_degrees<uint32_t>(graph_info, partitioner, *cache);

  ASSERT_EQ(degrees[0],  19);
  ASSERT_EQ(degrees[99], 59);
  ASSERT_EQ(degrees[28], 19);
}

TEST(GraphOps, CountSparseOutDegreeUnDirected) {
  init_cluster_info();

  using part_spec_t = plato::hash_by_source_t<>;

  graph_info_t graph_info;
  graph_info.is_directed_ = false;

  auto cache = load_edges_cache<empty_t>(&graph_info, "data/graph/v100_e2150_ua_c3.csv", edge_format_t::CSV, dummy_decoder<empty_t>);

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  auto degrees = generate_sparse_out_degrees<uint32_t>(graph_info, partitioner, *cache);

  ASSERT_EQ(degrees[0],  19);
  ASSERT_EQ(degrees[99], 59);
  ASSERT_EQ(degrees[28], 19);
}

int main(int argc, char** argv) {
  // Filter out Google Test arguments
  ::testing::InitGoogleTest(&argc, argv);

  google::InitGoogleLogging("graphkit-test");
  google::LogToStderr();

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  // set OpenMP if not set
  if (nullptr == getenv("OMP_NUM_THREADS")) {
    setenv("OMP_NUM_THREADS", "3", 1);
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


