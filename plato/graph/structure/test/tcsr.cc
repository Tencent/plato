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

#include "plato/graph/structure/tcsr.hpp"

#include <set>
#include <memory>
#include <unordered_set>

#include "mpi.h"
#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gflags/gflags.h"

#include "plato/util/gtest_mpi_listener.hpp"
#include "plato/graph/structure.hpp"
#include "plato/graph/partition/hash.hpp"
#include "plato/graph/partition/sequence.hpp"

void init_cluster_info(void) {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  cluster_info.partitions_   = 1;
  cluster_info.partition_id_ = 0;
  cluster_info.threads_      = 3;
  cluster_info.sockets_      = 1;
}

namespace plato {

inline bool operator==(const plato::edge_unit_t<plato::empty_t>& lhs, const plato::edge_unit_t<plato::empty_t>& rhs) {
  return (lhs.src_ == rhs.src_) && (lhs.dst_ == rhs.dst_);
}

template <typename T>
inline bool operator==(const plato::edge_unit_t<T>& lhs, const plato::edge_unit_t<T>& rhs) {
  return (lhs.src_ == rhs.src_) && (lhs.dst_ == rhs.dst_) && (lhs.edata_ == rhs.edata_);
}

}

namespace std {

template <>
struct hash<plato::edge_unit_t<plato::empty_t>> {
  size_t operator()(const plato::edge_unit_t<plato::empty_t>& k) const {
    return (k.src_) | ((size_t)k.dst_ << 32);
  }
};

}

TEST(TCSR, Init) {
  init_cluster_info();

  using part_spec_t = plato::hash_by_source_t<>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::tcsr_t<plato::empty_t, plato::empty_t, part_spec_t> storage(100, partitioner);
}

TEST(TCSR, QueryNeighbours) {
  init_cluster_info();

  using part_spec_t          = plato::hash_by_source_t<>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::tcsr_t<plato::empty_t, plato::empty_t, part_spec_t> storage(200, partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto pcache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/v100_e2150_ua_c3.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);
  auto& cache = *pcache;

  ASSERT_EQ(0, storage.load_edges_from_cache(graph_info, cache));

  adj_unit_list_spec_t neighbours = storage.neighbours(0);
  ASSERT_EQ(19, neighbours.end_ - neighbours.begin_);

  std::set<plato::vid_t> distinct_v;
  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    distinct_v.insert(it->neighbour_);
  }

  ASSERT_EQ(19, distinct_v.size());
  for (auto& v: distinct_v) {
    ASSERT_TRUE(v >= 1 && v <= 19);
  }

  neighbours = storage.neighbours(10);
  ASSERT_EQ(9, neighbours.end_ - neighbours.begin_);

  distinct_v.clear();
  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    distinct_v.insert(it->neighbour_);
  }

  ASSERT_EQ(9, distinct_v.size());
  for (auto& v: distinct_v) {
    ASSERT_TRUE(v >= 11 && v <= 19);
  }

  neighbours = storage.neighbours(98);
  ASSERT_EQ(1, neighbours.end_ - neighbours.begin_);

  distinct_v.clear();
  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    distinct_v.insert(it->neighbour_);
  }

  ASSERT_EQ(1, distinct_v.size());
  for (auto& v: distinct_v) {
    ASSERT_TRUE(v >= 99 && v <= 99);
  }
}

TEST(TCSR, LoadFromCacheWeightlessUndirected) {
  init_cluster_info();

  using part_spec_t          = plato::hash_by_source_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<plato::empty_t>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::tcsr_t<plato::empty_t, plato::empty_t, part_spec_t> storage(200, partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = false;

  auto pcache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/v100_e2150_ua_c3.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);
  auto& cache = *pcache;

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache.size(); ++e_i) {
    edge_unit_spec_t edge = cache[e_i];

    edges.emplace_back(edge);
    auto tmp = edge.src_; edge.src_ = edge.dst_; edge.dst_ = tmp;
    edges.emplace_back(edge);
  }

  ASSERT_EQ(0, storage.load_edges_from_cache(graph_info, cache));

  plato::vid_t v_count = 0;
  plato::vid_t e_count = 0;
  auto traversal = [&] (plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
    __sync_fetch_and_add(&v_count, 1);

    for (auto* padj = adjs.begin_; padj != adjs.end_; ++padj) {
      __sync_fetch_and_add(&e_count, 1);

      [&](void) {
        ASSERT_THAT(edges, testing::Contains(edge_unit_spec_t { v_i, padj->neighbour_ }));
      }();
    }
    return true;
  };

  storage.reset_traversal();

  #pragma omp parallel
  {
    size_t chunk_size = 1;
    while (storage.next_chunk(traversal, &chunk_size)) { }
  }

  ASSERT_EQ(v_count, 100);
  ASSERT_EQ(e_count, 4300);
}

TEST(TCSR, LoadFromCacheWeightlessDirected) {
  init_cluster_info();

  using part_spec_t          = plato::hash_by_source_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<plato::empty_t>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::tcsr_t<plato::empty_t, plato::empty_t, part_spec_t> storage(200, partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto pcache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/v100_e4300_da_c3.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);
  auto& cache = *pcache;

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache.size(); ++e_i) {
    edges.emplace_back(cache[e_i]);
  }

  ASSERT_EQ(0, storage.load_edges_from_cache(graph_info, cache));

  plato::vid_t v_count = 0;
  plato::vid_t e_count = 0;
  auto traversal = [&](plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
    __sync_fetch_and_add(&v_count, 1);

    for (auto* padj = adjs.begin_; padj != adjs.end_; ++padj) {
      __sync_fetch_and_add(&e_count, 1);

      [&](void) {
        ASSERT_THAT(edges, testing::Contains(edge_unit_spec_t { v_i, padj->neighbour_ }));
      }();
    }
    return true;
  };

  storage.reset_traversal();

  #pragma omp parallel
  {
    size_t chunk_size = 1;
    while (storage.next_chunk(traversal, &chunk_size)) { }
  }

  ASSERT_EQ(v_count, 100);
  ASSERT_EQ(e_count, 4300);
}

TEST(TCSR, LoadFromCacheWeightedDirected) {
  init_cluster_info();

  using part_spec_t          = plato::hash_by_source_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<float>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<float>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::tcsr_t<float, plato::empty_t, part_spec_t> storage(200, partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto pcache = plato::load_edges_cache<float>(&graph_info, "data/graph/graph_10_9.csv",
      plato::edge_format_t::CSV, plato::float_decoder);
  auto& cache = *pcache;

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache.size(); ++e_i) {
    edges.emplace_back(cache[e_i]);
  }

  ASSERT_EQ(0, storage.load_edges_from_cache(graph_info, cache));

  plato::vid_t v_count = 0;
  plato::vid_t e_count = 0;
  auto traversal = [&](plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
    __sync_fetch_and_add(&v_count, 1);

    for (auto* padj = adjs.begin_; padj != adjs.end_; ++padj) {
      __sync_fetch_and_add(&e_count, 1);

      [&](void) {
        ASSERT_THAT(edges, testing::Contains(
          edge_unit_spec_t { v_i, padj->neighbour_ , padj->edata_}
        ));
      }();
    }
    return true;
  };

  storage.reset_traversal();

  #pragma omp parallel
  {
    size_t chunk_size = 1;
    while (storage.next_chunk(traversal, &chunk_size)) { }
  }

  ASSERT_EQ(v_count, 9);
  ASSERT_EQ(e_count, 9);
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



