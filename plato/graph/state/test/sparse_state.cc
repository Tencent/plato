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

#include "plato/graph/state/sparse_state.hpp"

#include <set>
#include <map>
#include <mutex>
#include <vector>

#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gtest_mpi_listener.hpp"

#include "plato/util/bitmap.hpp"
#include "plato/graph/partition/hash.hpp"
#include "plato/graph/partition/sequence.hpp"

void init_cluster_info(void) {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  cluster_info.partitions_   = 1;
  cluster_info.partition_id_ = 0;
  cluster_info.threads_      = 3;
  cluster_info.sockets_      = 1;
}

TEST(SparseState, Init) {
  init_cluster_info();

  using part_spec_t = plato::hash_by_source_t<>;
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::sparse_state_t<uint32_t, part_spec_t> state(1024, partitioner);
}

TEST(SparseState, UpsertValue) {
  init_cluster_info();

  using part_spec_t = plato::hash_by_source_t<>;
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::sparse_state_t<uint32_t, part_spec_t> state(1024, partitioner);

  auto dummy = [](uint32_t&) {};

  state.upsert(0, dummy, 1);
  state.upsert(1, dummy, 2);

  state.lock();
  ASSERT_EQ(1, state[0]);
  ASSERT_EQ(2, state[1]);
  state.unlock();

  state.upsert(2, dummy, 3);
  state.upsert(3, dummy, 4);

  state.lock();
  ASSERT_EQ(1, state[0]);
  ASSERT_EQ(2, state[1]);
  ASSERT_EQ(3, state[2]);
  ASSERT_EQ(4, state[3]);
}

TEST(SparseState, TravseWithoutActive) {
  init_cluster_info();

  using part_spec_t = plato::hash_by_source_t<>;
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::sparse_state_t<uint32_t, part_spec_t> state(5000, partitioner);

  const size_t length = 1007;
  for (size_t i = 0; i < length; ++i) {
    state.upsert(i, [](uint32_t&){ }, i);
  }

  std::mutex mtx;
  std::map<plato::vid_t, uint32_t> values;

  state.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 1;
    while (state.next_chunk([&](plato::vid_t v_i, uint32_t* pval) {
      mtx.lock();
      values.emplace(v_i, *pval);
      mtx.unlock();
    }, &chunk_size)) { }
  }

  ASSERT_EQ(values.size(), length);
  for (size_t i = 0; i < length; ++i) {
    ASSERT_THAT(values, testing::Contains(std::make_pair<plato::vid_t, uint32_t>(i, i)));
  }
}

TEST(SparseState, TravseWithActive) {
  init_cluster_info();

  const plato::vid_t max_v_i = 5000;
  using part_spec_t = plato::hash_by_source_t<>;
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::sparse_state_t<uint32_t, part_spec_t> state(max_v_i, partitioner);

  const size_t length = 1007;
  for (size_t i = 0; i < length; ++i) {
    state.upsert(i, [](uint32_t&){ }, i);
  }

  std::mutex mtx;
  std::map<plato::vid_t, uint32_t> values;
  std::shared_ptr<plato::bitmap_t<>> bitmap(new plato::bitmap_t<>(max_v_i + 1));

  bitmap->set_bit(0);
  bitmap->set_bit(5);
  bitmap->set_bit(20);
  bitmap->set_bit(1006);
  bitmap->set_bit(1007);  // not-existed

  state.reset_traversal(bitmap);
  #pragma omp parallel
  {
    size_t chunk_size = 1;
    while (state.next_chunk([&](plato::vid_t v_i, uint32_t* pval) {
      mtx.lock();
      values.emplace(v_i, *pval);
      mtx.unlock();
    }, &chunk_size)) { }
  }

  ASSERT_EQ(values.size(), 4);

  ASSERT_THAT(values, testing::Contains(std::make_pair<plato::vid_t, uint32_t>(0, 0)));
  ASSERT_THAT(values, testing::Contains(std::make_pair<plato::vid_t, uint32_t>(5, 5)));
  ASSERT_THAT(values, testing::Contains(std::make_pair<plato::vid_t, uint32_t>(20, 20)));
  ASSERT_THAT(values, testing::Contains(std::make_pair<plato::vid_t, uint32_t>(1006, 1006)));
  ASSERT_THAT(values, testing::Not(testing::Contains(std::make_pair<plato::vid_t, uint32_t>(1007, 1007))));
}

TEST(SparseState, ForEachWithActive) {
  init_cluster_info();

  const plato::vid_t max_v_i = 5000;
  using part_spec_t = plato::hash_by_source_t<>;
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  plato::sparse_state_t<uint32_t, part_spec_t> state(max_v_i, partitioner);

  const size_t length = 1007;
  for (size_t i = 0; i < length; ++i) {
    state.upsert(i, [](uint32_t&){ }, i);
  }

  std::mutex mtx;
  std::map<plato::vid_t, uint32_t> values;
  std::shared_ptr<plato::bitmap_t<>> bitmap(new plato::bitmap_t<>(max_v_i + 1));

  bitmap->set_bit(0);
  bitmap->set_bit(5);
  bitmap->set_bit(20);
  bitmap->set_bit(1006);
  bitmap->set_bit(1007);  // not-existed

  uint32_t sum = state.foreach<uint32_t>([&](plato::vid_t v_i, uint32_t* pval) {
    mtx.lock();
    values.emplace(v_i, *pval);
    mtx.unlock();
    return *pval;
  }, bitmap.get());

  ASSERT_EQ(sum, 1031);
  ASSERT_EQ(values.size(), 4);

  ASSERT_THAT(values, testing::Contains(std::make_pair<plato::vid_t, uint32_t>(0, 0)));
  ASSERT_THAT(values, testing::Contains(std::make_pair<plato::vid_t, uint32_t>(5, 5)));
  ASSERT_THAT(values, testing::Contains(std::make_pair<plato::vid_t, uint32_t>(20, 20)));
  ASSERT_THAT(values, testing::Contains(std::make_pair<plato::vid_t, uint32_t>(1006, 1006)));
  ASSERT_THAT(values, testing::Not(testing::Contains(std::make_pair<plato::vid_t, uint32_t>(1007, 1007))));
}

int main(int argc, char** argv) {
  // Filter out Google Test arguments
  ::testing::InitGoogleTest(&argc, argv);

  google::InitGoogleLogging("plato-test");
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


