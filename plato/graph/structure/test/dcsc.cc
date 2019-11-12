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

#include "plato/graph/structure/dcsc.hpp"

#include <memory>
#include <unordered_set>

#include "mpi.h"
#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gflags/gflags.h"
#include "gtest_mpi_listener.hpp"

#include "plato/graph/structure.hpp"
#include "plato/graph/structure/bcsr.hpp"
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

TEST(DCSC, Init) {
  init_cluster_info();

  using part_spec_t = plato::hash_by_destination_t<>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::bcsr_t<plato::empty_t, part_spec_t> storage(partitioner);
}

TEST(DCSC, LoadFromCacheWeightlessUndirected) {
  init_cluster_info();

  using part_spec_t          = plato::hash_by_destination_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<plato::empty_t>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::dcsc_t<plato::empty_t, part_spec_t> storage(partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = false;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/v100_e2150_ua_c3.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edge_unit_spec_t edge = (*cache)[e_i];

    edges.emplace_back(edge);
    auto tmp = edge.src_; edge.src_ = edge.dst_; edge.dst_ = tmp;
    edges.emplace_back(edge);
  }

  ASSERT_EQ(0, storage.load_from_cache(graph_info, *cache));

  ASSERT_EQ(storage.vertices(), 100);
  ASSERT_EQ(storage.edges(),    4300);

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
    size_t chunk_size = 15;
    while (storage.next_chunk(traversal, &chunk_size)) { }
  }

  ASSERT_EQ(v_count, 100);
  ASSERT_EQ(e_count, 4300);
}

TEST(DCSC, LoadFromCacheWeightedDirected) {
  init_cluster_info();

  using part_spec_t          = plato::hash_by_destination_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<float>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<float>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::dcsc_t<float, part_spec_t> storage(partitioner);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<float>(&graph_info, "data/graph/graph_10_9.csv",
      plato::edge_format_t::CSV, plato::float_decoder);

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edges.emplace_back((*cache)[e_i]);
  }

  ASSERT_EQ(0, storage.load_from_cache(graph_info, *cache));

  ASSERT_EQ(storage.vertices(), 9);
  ASSERT_EQ(storage.edges(),    9);

  plato::vid_t v_count = 0;
  plato::vid_t e_count = 0;
  auto traversal = [&](plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
    __sync_fetch_and_add(&v_count, 1);

    for (auto* padj = adjs.begin_; padj != adjs.end_; ++padj) {
      __sync_fetch_and_add(&e_count, 1);

      [&](void) {
        ASSERT_THAT(edges, testing::Contains(
            edge_unit_spec_t { padj->neighbour_, v_i, padj->edata_}
        ));
      }();
    }
    return true;
  };

  storage.reset_traversal();

  #pragma omp parallel
  {
    size_t chunk_size = 4;
    while (storage.next_chunk(traversal, &chunk_size)) { }
  }

  ASSERT_EQ(v_count, 9);
  ASSERT_EQ(e_count, 9);
}

TEST(DCSC, LoadFromBCSRWeightlessUndirected) {
  init_cluster_info();

  using part_bcsr_spec_t     = plato::hash_by_source_t<>;
  using part_dcsc_spec_t     = plato::hash_by_destination_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<plato::empty_t>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_bcsr_spec_t> part_bcsr(new part_bcsr_spec_t());
  std::shared_ptr<part_dcsc_spec_t> part_dcsc(new part_dcsc_spec_t());

  plato::bcsr_t<plato::empty_t, part_bcsr_spec_t> bcsr(part_bcsr);
  plato::dcsc_t<plato::empty_t, part_dcsc_spec_t> dcsc(part_dcsc);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = false;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/v100_e2150_ua_c3.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edge_unit_spec_t edge = (*cache)[e_i];

    edges.emplace_back(edge);
    auto tmp = edge.src_; edge.src_ = edge.dst_; edge.dst_ = tmp;
    edges.emplace_back(edge);
  }

  ASSERT_EQ(0, bcsr.load_from_cache(graph_info, *cache));
  ASSERT_EQ(0, dcsc.load_from_graph(graph_info, bcsr, true));

  ASSERT_EQ(dcsc.vertices(), 100);
  ASSERT_EQ(dcsc.edges(),    4300);

  plato::vid_t v_count = 0;
  plato::vid_t e_count = 0;
  auto traversal = [&](plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
    __sync_fetch_and_add(&v_count, 1);

    for (auto* padj = adjs.begin_; padj != adjs.end_; ++padj) {
      __sync_fetch_and_add(&e_count, 1);

      [&](void) {
        ASSERT_THAT(edges, testing::Contains(edge_unit_spec_t { padj->neighbour_, v_i }));
      }();
    }
    return true;
  };

  dcsc.reset_traversal();

  #pragma omp parallel
  {
    size_t chunk_size = 15;
    while (dcsc.next_chunk(traversal, &chunk_size)) { }
  }

  ASSERT_EQ(v_count, 100);
  ASSERT_EQ(e_count, 4300);
}

TEST(DCSC, LoadFromBCSRWeightlessDirected) {
  init_cluster_info();

  using part_bcsr_spec_t     = plato::hash_by_source_t<>;
  using part_dcsc_spec_t     = plato::hash_by_destination_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<plato::empty_t>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  std::shared_ptr<part_bcsr_spec_t> part_bcsr(new part_bcsr_spec_t());
  std::shared_ptr<part_dcsc_spec_t> part_dcsc(new part_dcsc_spec_t());

  plato::bcsr_t<plato::empty_t, part_bcsr_spec_t> bcsr(part_bcsr);
  plato::dcsc_t<plato::empty_t, part_dcsc_spec_t> dcsc(part_dcsc);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/v100_e4300_da_c3.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edges.emplace_back((*cache)[e_i]);
  }

  ASSERT_EQ(0, bcsr.load_from_cache(graph_info, *cache));
  ASSERT_EQ(0, dcsc.load_from_graph(graph_info, bcsr, true));

  ASSERT_EQ(dcsc.vertices(), 100);
  ASSERT_EQ(dcsc.edges(),    4300);

  plato::vid_t v_count = 0;
  plato::vid_t e_count = 0;
  auto traversal = [&](plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
    __sync_fetch_and_add(&v_count, 1);

    for (auto* padj = adjs.begin_; padj != adjs.end_; ++padj) {
      __sync_fetch_and_add(&e_count, 1);

      [&](void) {
        ASSERT_THAT(edges, testing::Contains(edge_unit_spec_t { padj->neighbour_, v_i }));
      }();
    }
    return true;
  };

  dcsc.reset_traversal();

  #pragma omp parallel
  {
    size_t chunk_size = 15;
    while (dcsc.next_chunk(traversal, &chunk_size)) { }
  }

  ASSERT_EQ(v_count, 100);
  ASSERT_EQ(e_count, 4300);
}

TEST(DCSC, LoadFromBCSRWeightedDirected) {
  init_cluster_info();

  using part_bcsr_spec_t     = plato::hash_by_source_t<>;
  using part_dcsc_spec_t     = plato::hash_by_destination_t<>;
  using edge_unit_spec_t     = plato::edge_unit_t<float>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<float>;

  std::shared_ptr<part_bcsr_spec_t> part_bcsr(new part_bcsr_spec_t());
  std::shared_ptr<part_dcsc_spec_t> part_dcsc(new part_dcsc_spec_t());

  plato::bcsr_t<float, part_bcsr_spec_t> bcsr(part_bcsr);
  plato::dcsc_t<float, part_dcsc_spec_t> dcsc(part_dcsc);

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<float>(&graph_info, "data/graph/graph_10_9.csv",
      plato::edge_format_t::CSV, plato::float_decoder);

  std::vector<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edges.emplace_back((*cache)[e_i]);
  }

  ASSERT_EQ(0, bcsr.load_from_cache(graph_info, *cache));
  ASSERT_EQ(0, dcsc.load_from_graph(graph_info, bcsr, true));

  ASSERT_EQ(dcsc.vertices(), 9);
  ASSERT_EQ(dcsc.edges(),    9);

  plato::vid_t v_count = 0;
  plato::vid_t e_count = 0;
  auto traversal = [&](plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
    __sync_fetch_and_add(&v_count, 1);

    for (auto* padj = adjs.begin_; padj != adjs.end_; ++padj) {
      __sync_fetch_and_add(&e_count, 1);

      [&](void) {
        ASSERT_THAT(edges, testing::Contains(edge_unit_spec_t { padj->neighbour_, v_i, padj->edata_ }));
      }();
    }
    return true;
  };

  dcsc.reset_traversal();

  #pragma omp parallel
  {
    size_t chunk_size = 15;
    while (dcsc.next_chunk(traversal, &chunk_size)) { }
  }

  ASSERT_EQ(v_count, 9);
  ASSERT_EQ(e_count, 9);
}

void TestTraverse(const plato::traverse_opts_t& opts) {
  init_cluster_info();

  using part_spec_t          = plato::sequence_balanced_by_source_t;
  using edge_unit_spec_t     = plato::edge_unit_t<plato::empty_t>;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<plato::empty_t>;

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/v100_e4300_da_c3.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  std::vector<uint32_t> degrees = plato::generate_dense_out_degrees<uint32_t>(graph_info, *cache);

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t(degrees.data(), graph_info.vertices_, graph_info.edges_));
  plato::dcsc_t<plato::empty_t, part_spec_t> storage(partitioner);

  std::unordered_set<edge_unit_spec_t> edges;
  for (size_t e_i = 0; e_i < cache->size(); ++e_i) {
    edges.emplace((*cache)[e_i]);
  }

  ASSERT_EQ(0, storage.load_from_cache(graph_info, *cache));

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

TEST(DCSC, TraverseOrigin) {
  plato::traverse_opts_t opts;
  opts.mode_ = plato::traverse_mode_t::ORIGIN;
  TestTraverse(opts);
}

TEST(DCSC, TraverseRandom) {
  plato::traverse_opts_t opts;
  opts.mode_ = plato::traverse_mode_t::RANDOM;
  TestTraverse(opts);
}

TEST(DCSC, TraverseCircle) {
  plato::traverse_opts_t opts;
  opts.mode_ = plato::traverse_mode_t::CIRCLE;
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



