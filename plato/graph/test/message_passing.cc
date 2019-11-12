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

#include "plato/graph/message_passing.hpp"

#include <mutex>
#include <vector>
#include <type_traits>

#include "mpi.h"
#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gflags/gflags.h"
#include "gtest_mpi_listener.hpp"

#include "plato/util/spinlock.hpp"
#include "plato/graph/state.hpp"
#include "plato/graph/structure.hpp"

void init_cluster_info(void) {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  cluster_info.partitions_   = 1;
  cluster_info.partition_id_ = 0;
  cluster_info.threads_      = 3;
  cluster_info.sockets_      = 1;
}

TEST(MessagePassing, AggregateMessage) {
  init_cluster_info();

  using context_spec_t = plato::mepa_ag_context_t<uint32_t>;
  using message_spec_t = plato::mepa_ag_message_t<uint32_t>;

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto pdcsc = plato::create_dcsc_seqs_from_path<float>(
    &graph_info, "data/graph/graph_10_9.csv",
    plato::edge_format_t::CSV, plato::float_decoder
  );

  using graph_spec_t = std::remove_reference<decltype(*pdcsc)>::type;
  using adj_unit_list_spec_t = graph_spec_t::adj_unit_list_spec_t;

  std::vector<int> values(10, 0);
  std::vector<plato::spinlock_t> locks(10);

  int sum = plato::aggregate_message<uint32_t, int, graph_spec_t>(*pdcsc,
    [&](const context_spec_t& context, plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
      for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
        context.send(message_spec_t { v_i, 1 });
      }
    },
    [&](int /*p_i*/, message_spec_t& msg) {
      locks[msg.v_i_].lock();
      values[msg.v_i_] += msg.message_;
      locks[msg.v_i_].unlock();
      return 1;
    }
  );

  ASSERT_EQ(9, sum);

  ASSERT_EQ(0, values[0]);
  for (size_t i = 1; i < values.size(); ++i) {
    ASSERT_EQ(1, values[i]);
  }
}

TEST(MessagePassing, SpreadMessageWithoutActive) {
  init_cluster_info();

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto pbcsr = plato::create_bcsr_seqd_from_path<float>(
    &graph_info, "data/graph/graph_10_9.csv",
    plato::edge_format_t::CSV, plato::float_decoder
  );

  using context_spec_t = plato::mepa_sd_context_t<uint32_t>;

  plato::bitmap_t<> bitmap(graph_info.vertices_);
  bitmap.fill();

  std::vector<uint32_t> values(10, 0);
  std::vector<plato::spinlock_t> locks(10);

  int sum = plato::spread_message<plato::vid_t, int>(bitmap,
    [&](const context_spec_t& context, plato::vid_t v_i) {
      context.send(0, v_i);
    },
    [&](plato::vid_t& v_i) {
      locks[v_i].lock();
      values[v_i] += 1;
      locks[v_i].unlock();
      return 1;
    }
  );

  ASSERT_EQ(10, sum);
  ASSERT_THAT(values, testing::Each(testing::Eq(1)));
}

TEST(MessagePassing, SpreadMessageWithActive) {
  init_cluster_info();

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto pbcsr = plato::create_bcsr_seqd_from_path<float>(
    &graph_info, "data/graph/graph_10_9.csv",
    plato::edge_format_t::CSV, plato::float_decoder
  );

  using context_spec_t = plato::mepa_sd_context_t<uint32_t>;

  plato::bitmap_t<> bitmap(graph_info.vertices_);
  bitmap.set_bit(0);
  bitmap.set_bit(5);
  bitmap.set_bit(9);

  std::vector<uint32_t> values(10, 0);
  std::vector<plato::spinlock_t> locks(10);

  int sum = plato::spread_message<plato::vid_t, int>(bitmap,
    [&](const context_spec_t& context, plato::vid_t v_i) {
      context.send(0, v_i);
    },
    [&](plato::vid_t& v_i) {
      locks[v_i].lock();
      values[v_i] += 1;
      locks[v_i].unlock();
      return 1;
    }
  );

  ASSERT_EQ(3, sum);

  ASSERT_EQ(1, values[0]);
  ASSERT_EQ(1, values[5]);
  ASSERT_EQ(1, values[9]);

  int __sum = 0;
  for (const auto& v: values) {
    __sum += v;
  }
  ASSERT_EQ(3, __sum);
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

