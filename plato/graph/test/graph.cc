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

#include "plato/graph/graph.hpp"

#include "mpi.h"
#include "omp.h"
#include "gtest/gtest.h"
#include "gflags/gflags.h"
#include "gtest_mpi_listener.hpp"

using namespace plato;

void init_graph_info(void) {
  cluster_info_t& pcinfo = cluster_info_t::get_instance();

  pcinfo.partitions_   = 1;
  pcinfo.partition_id_ = 0;
  pcinfo.threads_      = 3;
  pcinfo.sockets_      = 1;
}

TEST(Graph, CreateGraphFromEdges) {
  init_graph_info();

  auto graph = create_graph_from_edges<empty_t, adjlist_t<empty_t>, hash_by_source_t<>>("data/graph/v100_e4300_da_c3.csv",
      edge_format_t::CSV, dummy_decoder<empty_t>, false);
  ASSERT_NE(nullptr, graph);
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

