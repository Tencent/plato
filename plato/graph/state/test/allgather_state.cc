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

#include "plato/graph/state/allgather_state.hpp"

#include <map>
#include <atomic>
#include <vector>

#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gflags/gflags.h"
#include "glog/stl_logging.h"
#include "gtest_mpi_listener.hpp"
#include "yas/types/std/vector.hpp"

#include "plato/util/spinlock.hpp"

void init_cluster_info(void) {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  cluster_info.partitions_   = 1;
  cluster_info.partition_id_ = 0;
  cluster_info.threads_      = 3;
  cluster_info.sockets_      = 1;
}


TEST(Parallel, AllgatherStateTrivial) {
  init_cluster_info();

  plato::allgather_state_opts_t opt;
  opt.threads_         = -1;
  init_cluster_info();
  auto& cluster_info = plato::cluster_info_t::get_instance();

  using vid_t = uint32_t;

  plato::graph_info_t graph_info; 
  graph_info.max_v_i_ = 1024;
  using part_spec_t = plato::sequence_balanced_by_destination_t;
  std::vector<vid_t> offset(2);
  offset[0] = 0;
  offset[1] = graph_info.max_v_i_;
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t(offset));
  plato::dense_state_t<vid_t, part_spec_t> state(graph_info.max_v_i_, partitioner);

  state.fill(cluster_info.partition_id_);

  plato::allgather_state<vid_t>(state, opt);

  int partitions = cluster_info.partitions_;

  for (int p = 0; p < partitions; p ++) {
    vid_t v_begin = partitioner->offset_[p];
    vid_t v_end = partitioner->offset_[p+1];
    for (vid_t vit = v_begin; vit < v_end; vit ++) {
      ASSERT_EQ(state[vit], p);
    }
  }
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

