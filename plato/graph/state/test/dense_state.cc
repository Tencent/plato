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

#include "plato/graph/state/dense_state.hpp"

#include <set>
#include <mutex>
#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gtest_mpi_listener.hpp"

#include "plato/util/bitmap.hpp"
#include "plato/util/mmap_alloc.hpp"
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

TEST(DenseState, Init) {
  init_cluster_info();

  using part_spec_t = plato::hash_by_source_t<>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());

  plato::graph_info_t graph_info; graph_info.max_v_i_ = 1024;
  plato::dense_state_t<uint32_t, part_spec_t> state(graph_info.max_v_i_, partitioner);
}

TEST(DenseState, TravseSeqPartitionWithoutActive) {
  init_cluster_info();

  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.partitions_   = 3;
  cluster_info.partition_id_ = 1;

  using part_spec_t = plato::sequence_balanced_by_source_t;

  const plato::vid_t max_v_i = 49;
  std::vector<plato::vid_t> offset({ 0, 5, 44, max_v_i + 1 });
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t(offset));

  plato::graph_info_t graph_info; graph_info.max_v_i_ = max_v_i;
  plato::dense_state_t<uint32_t, part_spec_t> state(graph_info.max_v_i_, partitioner);

  std::mutex mtx;
  std::set<plato::vid_t> seen_v;

  state.reset_traversal();

  size_t chunk_size = 3;

  #pragma omp parallel
  {
    while (state.next_chunk(
      [&](plato::vid_t v_i, uint32_t* pval) {
        mtx.lock();
        seen_v.emplace(v_i);
        mtx.unlock();
        return true;
      }, &chunk_size)) { }
  }

  ASSERT_EQ(seen_v.size(), offset[cluster_info.partition_id_ + 1] - offset[cluster_info.partition_id_]);
  for (plato::vid_t v_i = offset[cluster_info.partition_id_]; v_i < offset[cluster_info.partition_id_ + 1]; ++v_i) {
    ASSERT_THAT(seen_v, testing::Contains(v_i));
  }
}

TEST(DenseState, TravseSeqPartitionRandomWithoutActive) {
  init_cluster_info();

  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.partitions_   = 3;
  cluster_info.partition_id_ = 1;

  using part_spec_t = plato::sequence_balanced_by_source_t;

  const plato::vid_t max_v_i = 3012;
  std::vector<plato::vid_t> offset({ 0, 1000, 2000, max_v_i + 1 });
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t(offset));

  plato::graph_info_t graph_info; graph_info.max_v_i_ = max_v_i;
  plato::dense_state_t<uint32_t, part_spec_t> state(graph_info.max_v_i_, partitioner);

  std::mutex mtx;
  size_t count = 0;
  std::set<plato::vid_t> seen_v;

  plato::traverse_opts_t opts; opts.mode_ = plato::traverse_mode_t::RANDOM;
  state.reset_traversal(nullptr, opts);

  size_t chunk_size = 1;
  #pragma omp parallel
  {
    while (state.next_chunk(
      [&](plato::vid_t v_i, uint32_t* pval) {
        mtx.lock();
        seen_v.emplace(v_i);
        ++count;
        mtx.unlock();
        return true;
      }, &chunk_size)) { }
  }

  ASSERT_EQ(seen_v.size(), offset[cluster_info.partition_id_ + 1] - offset[cluster_info.partition_id_]);
  ASSERT_EQ(seen_v.size(), count);
  for (plato::vid_t v_i = offset[cluster_info.partition_id_]; v_i < offset[cluster_info.partition_id_ + 1]; ++v_i) {
    ASSERT_THAT(seen_v, testing::Contains(v_i));
  }
}

TEST(DenseState, TravseSeqPartitionWithActive) {
  init_cluster_info();

  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.partitions_   = 3;
  cluster_info.partition_id_ = 1;

  using part_spec_t = plato::sequence_balanced_by_source_t;

  const plato::vid_t max_v_i = 257;
  std::vector<plato::vid_t> offset({ 0, 5, 134, max_v_i + 1 });
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t(offset));

  plato::graph_info_t graph_info; graph_info.max_v_i_ = max_v_i;
  plato::dense_state_t<uint32_t, part_spec_t> state(graph_info.max_v_i_, partitioner);

  std::mutex mtx;
  std::set<plato::vid_t> seen_v;
  std::shared_ptr<plato::bitmap_t<>> bitmap(new plato::bitmap_t<>(max_v_i + 1));

  bitmap->set_bit(5);
  bitmap->set_bit(6);
  bitmap->set_bit(42);
  bitmap->set_bit(133);
  bitmap->set_bit(134);

  state.reset_traversal(bitmap);

  size_t chunk_size = 3;

  #pragma omp parallel
  {
    while (state.next_chunk(
      [&](plato::vid_t v_i, uint32_t* pval) {
        mtx.lock();
        seen_v.emplace(v_i);
        mtx.unlock();
        return true;
      }, &chunk_size)) { }
  }

  ASSERT_EQ(seen_v.size(), 4);
  ASSERT_TRUE(seen_v.count(5));
  ASSERT_TRUE(seen_v.count(6));
  ASSERT_TRUE(seen_v.count(42));
  ASSERT_TRUE(seen_v.count(133));
  ASSERT_FALSE(seen_v.count(134));
}

TEST(DenseState, TravseSeqPartitionRandomWithActive) {
  init_cluster_info();

  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.partitions_   = 3;
  cluster_info.partition_id_ = 1;

  using part_spec_t = plato::sequence_balanced_by_source_t;

  const plato::vid_t max_v_i = 3025;
  std::vector<plato::vid_t> offset({ 0, 1000, 2000, max_v_i + 1 });
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t(offset));

  plato::graph_info_t graph_info; graph_info.max_v_i_ = max_v_i;
  plato::dense_state_t<uint32_t, part_spec_t> state(graph_info.max_v_i_, partitioner);

  std::mutex mtx;
  size_t count = 0;
  std::set<plato::vid_t> seen_v;
  std::shared_ptr<plato::bitmap_t<>> bitmap(new plato::bitmap_t<>(max_v_i + 1));

  bitmap->set_bit(5);
  bitmap->set_bit(999);
  bitmap->set_bit(1000);
  bitmap->set_bit(1001);
  bitmap->set_bit(1210);
  bitmap->set_bit(1999);
  bitmap->set_bit(2000);

  plato::traverse_opts_t opts; opts.mode_ = plato::traverse_mode_t::RANDOM;
  state.reset_traversal(bitmap, opts);

  size_t chunk_size = 1;
  #pragma omp parallel
  {
    while (state.next_chunk(
      [&](plato::vid_t v_i, uint32_t* pval) {
        mtx.lock();
        seen_v.emplace(v_i);
        ++count;
        mtx.unlock();
        return true;
      }, &chunk_size)) { }
  }

  ASSERT_EQ(seen_v.size(), 4);
  ASSERT_EQ(seen_v.size(), count);

  ASSERT_TRUE(seen_v.count(1000));
  ASSERT_TRUE(seen_v.count(1001));
  ASSERT_TRUE(seen_v.count(1210));
  ASSERT_TRUE(seen_v.count(1999));

  ASSERT_FALSE(seen_v.count(5));
  ASSERT_FALSE(seen_v.count(2000));
}

TEST(DenseState, TravseHashPartitionWithoutActive) {
  init_cluster_info();

  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.partitions_   = 3;
  cluster_info.partition_id_ = 1;

  using part_spec_t = plato::hash_by_source_t<>;

  const plato::vid_t max_v_i = 49;
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());

  std::set<plato::vid_t> existed_v;
  for (plato::vid_t i = 0; i <= max_v_i; ++i) {
    if (partitioner->get_partition_id(i) == cluster_info.partition_id_) {
      existed_v.emplace(i);
    }
  }

  plato::graph_info_t graph_info; graph_info.max_v_i_ = max_v_i;
  plato::dense_state_t<uint32_t, part_spec_t> state(graph_info.max_v_i_, partitioner);

  std::mutex mtx;
  std::vector<plato::vid_t> seen_v;

  state.reset_traversal();

  size_t chunk_size = 3;

  #pragma omp parallel
  {
    while (state.next_chunk(
      [&](plato::vid_t v_i, uint32_t* pval) {
        mtx.lock();
        seen_v.emplace_back(v_i);
        mtx.unlock();
        return true;
      }, &chunk_size)) { }
  }

  ASSERT_EQ(seen_v.size(), existed_v.size());
  for (const auto& v: seen_v) {
    ASSERT_THAT(existed_v, testing::Contains(v));
  }
}

TEST(DenseState, TravseHashPartitionWithActive) {
  init_cluster_info();

  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.partitions_   = 3;
  cluster_info.partition_id_ = 1;

  using part_spec_t = plato::hash_by_source_t<>;

  const plato::vid_t max_v_i = 633;
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());

  std::set<plato::vid_t> existed_v;
  for (plato::vid_t i = 0; i <= max_v_i; ++i) {
    if (partitioner->get_partition_id(i) == cluster_info.partition_id_) {
      existed_v.emplace(i);
    }
  }

  plato::graph_info_t graph_info; graph_info.max_v_i_ = max_v_i;
  plato::dense_state_t<uint32_t, part_spec_t> state(graph_info.max_v_i_, partitioner);

  std::mutex mtx;
  std::vector<plato::vid_t> seen_v;
  std::shared_ptr<plato::bitmap_t<>> bitmap(new plato::bitmap_t<>(max_v_i + 1));

  {
    size_t count = 0;
    for (auto it = existed_v.begin(); it != existed_v.end(); ++it) {
      bitmap->set_bit(*it);
      if (++count >= 3) { break; }
    }
  }

  state.reset_traversal(bitmap);

  size_t chunk_size = 3;

  #pragma omp parallel
  {
    while (state.next_chunk(
      [&](plato::vid_t v_i, uint32_t* pval) {
        mtx.lock();
        seen_v.emplace_back(v_i);
        mtx.unlock();
        return true;
      }, &chunk_size)) { }
  }

  ASSERT_EQ(seen_v.size(), 3);
  for (const auto& v: seen_v) {
    ASSERT_THAT(existed_v, testing::Contains(v));
  }
}

TEST(DenseState, FillHashPartition) {
  init_cluster_info();

  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.partitions_ = 3;

  using part_spec_t = plato::hash_by_source_t<>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());

  plato::graph_info_t graph_info; graph_info.max_v_i_ = 1024;
  plato::dense_state_t<uint32_t, part_spec_t> state(graph_info.max_v_i_, partitioner);

  state.fill(777);

  size_t chunk_size = 1024;

  state.reset_traversal();
  while (state.next_chunk(
    [&](plato::vid_t v_i, uint32_t* pval) {
      [&]() {
        ASSERT_EQ(*pval, 777);
      }();
      return true;
    }, &chunk_size)) { }
}

TEST(DenseState, FillSeqPartition) {
  init_cluster_info();

  auto& cluster_info = plato::cluster_info_t::get_instance();

  using part_spec_t = plato::sequence_balanced_by_source_t;

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/v100_e4300_da_c3.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  std::vector<plato::vid_t> degrees = plato::generate_dense_out_degrees<plato::vid_t>(graph_info, *cache);

  cluster_info.partitions_   = 3;
  cluster_info.partition_id_ = 2;

  std::shared_ptr<part_spec_t> part_impl (
      new part_spec_t(degrees.data(), graph_info.vertices_,
        graph_info.edges_, 200)
  );

  plato::dense_state_t<uint32_t, part_spec_t> state(graph_info.max_v_i_, part_impl);
  state.fill(777);

  size_t chunk_size = 1024;

  state.reset_traversal();
  while (state.next_chunk(
    [&](plato::vid_t v_i, uint32_t* pval) {
      [&]() {
        ASSERT_EQ(*pval, 777);
      }();
      return true;
    }, &chunk_size)) { }
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

