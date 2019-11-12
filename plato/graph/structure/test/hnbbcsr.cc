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

#include "plato/graph/structure/hnbbcsr.hpp"
#include <set>
#include <memory>
#include <vector>
#include <unordered_set>
#include "mpi.h"
#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gflags/gflags.h"
#include "gtest_mpi_listener.hpp"
#include "plato/graph/structure.hpp"
#include "plato/graph/partition/hash.hpp"
#include "plato/graph/partition/sequence.hpp"
#include "plato/util/bitmap.hpp"

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

TEST(HNBBCSR, Init) {
  
  init_cluster_info();

  using part_spec_t = plato::hash_by_source_t<>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::hnbbcsr_t<plato::empty_t, part_spec_t> storage(partitioner);
  
}

TEST(HNBBCSR, QueryNeighbours) {
  
  init_cluster_info();
  
  using part_spec_t          = plato::hash_by_source_t<>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::hnbbcsr_t<plato::empty_t, part_spec_t> storage(partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/non_coding_5_8.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  plato::sparse_state_t<uint32_t, part_spec_t> v_types(graph_info.vertices_, partitioner);

  plato::load_vertices_state_from_path<uint32_t>("data/graph/non_coding_5_8_vertic.csv", plato::edge_format_t::CSV,
      partitioner, plato::uint32_t_decoder, [&](plato::vertex_unit_t<uint32_t>&& unit) {
        v_types.insert(unit.vid_, unit.vdata_);
      });
  
  ASSERT_EQ(0, storage.load_from_cache(graph_info, *cache, v_types));
  
  storage.print_neighbours(1, 0);
  storage.print_neighbours(1, 1);
  storage.print_neighbours(2, 0);
  storage.print_neighbours(2, 1);
  storage.print_neighbours(100, 0);
  storage.print_neighbours(100, 1);
  storage.print_neighbours(1000, 0);
  storage.print_neighbours(1000, 1);
  storage.print_neighbours(1000000, 0);
  storage.print_neighbours(1000000, 1);

  {
    plato::traverse_opts_t opts;
    opts.mode_ = plato::traverse_mode_t::ORIGIN;
    storage.reset_traversal(opts);
  }
  size_t chunk_size = 2;
  storage.next_chunk([&](plato::vid_t v_i, const adj_unit_list_spec_t& ) {
    return true;
  }, &chunk_size);
  adj_unit_list_spec_t neighbours = storage.neighbours(100, 0);
  ASSERT_EQ(2, neighbours.end_ - neighbours.begin_);
  std::set<plato::vid_t> distinct_v;
  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    distinct_v.insert(it->neighbour_);
  }
  ASSERT_EQ(2, distinct_v.size());
  for (auto& v: distinct_v) {
    ASSERT_TRUE(v == 1 || v == 1000000);
  }

  neighbours = storage.neighbours(2, 0);
  ASSERT_EQ(1, neighbours.end_ - neighbours.begin_);
  distinct_v.clear();
  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    distinct_v.insert(it->neighbour_);
  }
  ASSERT_EQ(1, distinct_v.size());
  for (auto& v: distinct_v) {
    ASSERT_TRUE(v == 1000);
  }
  
  neighbours = storage.neighbours(1000000, 0);
  ASSERT_EQ(1, neighbours.end_ - neighbours.begin_);

  distinct_v.clear();
  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    distinct_v.insert(it->neighbour_);
  }
  
  ASSERT_EQ(1, distinct_v.size());
  for (auto& v: distinct_v) {
    ASSERT_TRUE(v == 1000);
  }
}

TEST(HNBBCSR, LoadFromCacheWeightlessUndirected) {
  init_cluster_info();

  using part_spec_t          = plato::hash_by_source_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<plato::empty_t>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::hnbbcsr_t<plato::empty_t, part_spec_t> storage(partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = false;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/non_coding_5_7.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  plato::sparse_state_t<uint32_t, part_spec_t> v_types(graph_info.vertices_, partitioner);
  plato::load_vertices_state_from_path<uint32_t>("data/graph/non_coding_5_8_vertic.csv", plato::edge_format_t::CSV,
      partitioner, plato::uint32_t_decoder, [&](plato::vertex_unit_t<uint32_t>&& unit) {
        v_types.insert(unit.vid_, unit.vdata_);
      });

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edge_unit_spec_t edge = (*cache)[e_i];

    edges.emplace_back(edge);
    auto tmp = edge.src_; edge.src_ = edge.dst_; edge.dst_ = tmp;
    edges.emplace_back(edge);
  }

  ASSERT_EQ(0, storage.load_from_cache(graph_info, *cache, v_types));
  ASSERT_EQ(storage.edges(), 14);
  ASSERT_EQ(storage.bitmap()->count(), 5);

  adj_unit_list_spec_t neighbours = storage.neighbours(100, 0);
  ASSERT_EQ(2, neighbours.end_ - neighbours.begin_);

  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    ASSERT_TRUE(it->neighbour_ == 1 || it->neighbour_ == 1000000);
  }
  
  neighbours = storage.neighbours(100, 1);

  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    ASSERT_TRUE(it->neighbour_ == 2);
  }
  
  neighbours = storage.neighbours(1000000, 1);
  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    ASSERT_TRUE(it->neighbour_ == 100);
  }
  
  neighbours = storage.neighbours(1000000, 0);
  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    ASSERT_TRUE(it->neighbour_ == 1000);
  }

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
  
  ASSERT_EQ(v_count, 5);
  ASSERT_EQ(e_count, 14);
}

TEST(HNBBCSR, LoadFromCacheWeightlessDirected) {
  init_cluster_info();

  using part_spec_t          = plato::hash_by_source_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<plato::empty_t>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::hnbbcsr_t<plato::empty_t, part_spec_t> storage(partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/non_coding_5_8.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  plato::sparse_state_t<uint32_t, part_spec_t> v_types(graph_info.vertices_, partitioner);
  plato::load_vertices_state_from_path<uint32_t>("data/graph/non_coding_5_8_vertic.csv", plato::edge_format_t::CSV,
      partitioner, plato::uint32_t_decoder, [&](plato::vertex_unit_t<uint32_t>&& unit) {
        v_types.insert(unit.vid_, unit.vdata_);
      });

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edges.emplace_back((*cache)[e_i]);
  }

  ASSERT_EQ(0, storage.load_from_cache(graph_info, *cache, v_types));

  //ASSERT_EQ(storage.vertices(), 100);
  ASSERT_EQ(storage.edges(), 8);
  ASSERT_EQ(storage.vertices(), 5);
  ASSERT_EQ(storage.bitmap()->count(), 5);

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

  ASSERT_EQ(v_count, 5);
  ASSERT_EQ(e_count, 8);
}

TEST(HNBBCSR, LoadFromCacheWeightedDirected) {
  init_cluster_info();

  using part_spec_t          = plato::hash_by_source_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<float>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<float>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::hnbbcsr_t<float, part_spec_t> storage(partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<float>(&graph_info, "data/graph/non_coding_5_8_weighted.csv",
      plato::edge_format_t::CSV, plato::float_decoder);

  plato::sparse_state_t<uint32_t, part_spec_t> v_types(graph_info.vertices_, partitioner);
  plato::load_vertices_state_from_path<uint32_t>("data/graph/non_coding_5_8_vertic.csv", plato::edge_format_t::CSV,
      partitioner, plato::uint32_t_decoder, [&](plato::vertex_unit_t<uint32_t>&& unit) {
        v_types.insert(unit.vid_, unit.vdata_);
      });

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edges.emplace_back((*cache)[e_i]);
  }

  ASSERT_EQ(0, storage.load_from_cache(graph_info, *cache, v_types));

  ASSERT_EQ(storage.vertices(), 5);
  ASSERT_EQ(storage.edges(), 8);
  ASSERT_EQ(storage.bitmap()->count(), 5);

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

  ASSERT_EQ(v_count, 5);
  ASSERT_EQ(e_count, 8);
}


TEST(HNBBCSR, LoadFromGraphWeightlessDirected) {
  init_cluster_info();

  using part_spec_t          = plato::hash_by_source_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<plato::empty_t>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::hnbbcsr_t<plato::empty_t, part_spec_t> storage(partitioner);
  plato::hnbbcsr_t<plato::empty_t, part_spec_t> storage_1(partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/non_coding_5_8.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);
  
  plato::sparse_state_t<uint32_t, part_spec_t> v_types(graph_info.vertices_, partitioner);
  plato::load_vertices_state_from_path<uint32_t>("data/graph/non_coding_5_8_vertic.csv", plato::edge_format_t::CSV,
      partitioner, plato::uint32_t_decoder, [&](plato::vertex_unit_t<uint32_t>&& unit) {
        v_types.insert(unit.vid_, unit.vdata_);
      });

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edges.emplace_back((*cache)[e_i]);
  }

  ASSERT_EQ(0, storage.load_from_cache(graph_info, *cache, v_types));
  ASSERT_EQ(0, storage_1.load_from_graph(graph_info, storage, true, v_types));

  //ASSERT_EQ(storage_1.vertices(), 100);
  ASSERT_EQ(storage_1.edges(), 8);
  ASSERT_EQ(storage_1.vertices(), 5);
  ASSERT_EQ(storage_1.bitmap()->count(), 5);

  adj_unit_list_spec_t neighbours = storage.neighbours(100, 0);
  ASSERT_EQ(2, neighbours.end_ - neighbours.begin_);

  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    ASSERT_TRUE(it->neighbour_ == 1 || it->neighbour_ == 1000000);
  }
  
  neighbours = storage.neighbours(100, 1);

  ASSERT_EQ(0, neighbours.end_ - neighbours.begin_);

  neighbours = storage.neighbours(1000000, 1);
  ASSERT_EQ(0, neighbours.end_ - neighbours.begin_);
  
  neighbours = storage.neighbours(1000000, 0);
  for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
    ASSERT_TRUE(it->neighbour_ == 1000);
  }

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

  storage_1.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 1;
    while (storage_1.next_chunk(traversal, &chunk_size)) { }
  }

  ASSERT_EQ(v_count, 5);
  ASSERT_EQ(e_count, 8);
}

void TestTraverse(const plato::traverse_opts_t& opts) {
  init_cluster_info();

  //using part_spec_t          = plato::sequence_balanced_by_destination_t;
  using part_spec_t          = plato::hash_by_source_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<plato::empty_t>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/non_coding_5_8.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  plato::sparse_state_t<uint32_t, part_spec_t> v_types(graph_info.vertices_, partitioner);
  plato::load_vertices_state_from_path<uint32_t>("data/graph/non_coding_5_8_vertic.csv", plato::edge_format_t::CSV,
      partitioner, plato::uint32_t_decoder, [&](plato::vertex_unit_t<uint32_t>&& unit) {
        v_types.insert(unit.vid_, unit.vdata_);
      });

  plato::hnbbcsr_t<plato::empty_t, part_spec_t> storage(partitioner);

  std::unordered_set<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edges.emplace((*cache)[e_i]);
  }

  ASSERT_EQ(0, storage.load_from_cache(graph_info, *cache, v_types));
  std::vector<edge_unit_spec_t> traverse_edges;
  auto traversal = [&](plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
    for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
      traverse_edges.emplace_back(edge_unit_spec_t { v_i, it->neighbour_ });
    }
    return true;
  };

  storage.reset_traversal(opts);

  size_t chunk_size = 1;
  while (storage.next_chunk(traversal, &chunk_size)) { }

  ASSERT_EQ(traverse_edges.size(), edges.size());
  for (const auto& edge: traverse_edges) {
    ASSERT_THAT(edges, testing::Contains(edge));
  }
}

TEST(HNBBCSR, TraverseOrigin) {
  plato::traverse_opts_t opts;
  opts.mode_ = plato::traverse_mode_t::ORIGIN;
  TestTraverse(opts);
}

TEST(HNBBCSR, TraverseRandom) {
  plato::traverse_opts_t opts;
  opts.mode_ = plato::traverse_mode_t::RANDOM;
  TestTraverse(opts);
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
