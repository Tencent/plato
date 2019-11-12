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

#include "plato/parallel/bsp.hpp"

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

struct nontrivial_t {
  nontrivial_t(void)
    : a_(0) { }

  nontrivial_t(uint64_t a, std::vector<float>&& b)
    : a_(a), b_(b) { }

  template<typename Ar>
  void serialize(Ar &ar) {
      ar & a_ & b_;
  }

  uint64_t           a_;
  std::vector<float> b_;
};

TEST(Parallel, BSPTrivial) {
  init_cluster_info();

  plato::bsp_opts_t opt;
  opt.threads_              = 3;
  opt.flying_send_per_node_ = 3;
  opt.flying_recv_          = 3;
  opt.global_size_          = 128;
  opt.local_capacity_       = 3;

  const int MSG_COUNT = 1033;

  plato::spinlock_t lock;
  std::map<int, int> num_count;

  auto bsp_send = [&](plato::bsp_send_callback_t<int> send) {
    int thread_id = omp_get_thread_num();
    int num_per_thread = MSG_COUNT / omp_get_num_threads();
    int start = num_per_thread * thread_id;
    int end   = (thread_id == (omp_get_num_threads() - 1)) ? MSG_COUNT : (start + num_per_thread);

    for (int i = start; i < end; ++i) {
      send(0, i);
    }
  };

  size_t recvd = 0;
  auto bsp_recv = [&](int p_i, plato::bsp_recv_pmsg_t<int>& pmsg) {
    __sync_fetch_and_add(&recvd, 1);

    lock.lock();
    num_count[*pmsg]++;
    lock.unlock();
  };
  ASSERT_EQ(0, plato::fine_grain_bsp<int>(bsp_send, bsp_recv, opt));

  for (int i = 0; i < MSG_COUNT; ++i) {
    ASSERT_THAT(num_count, testing::Contains(std::make_pair(i, 1)));
  }
}

TEST(Parallel, BSPNonTrivial) {
  init_cluster_info();

  plato::bsp_opts_t opt;
  opt.threads_              = 3;
  opt.flying_send_per_node_ = 3;
  opt.flying_recv_          = 3;
  opt.global_size_          = 33;
  opt.local_capacity_       = 3;

  const int MSG_COUNT = 1033;

  plato::spinlock_t lock;
  std::map<uint64_t, nontrivial_t> message_map;

  auto __send = [&](plato::bsp_send_callback_t<nontrivial_t> send) {
    int thread_id = omp_get_thread_num();
    int num_per_thread = MSG_COUNT / omp_get_num_threads();
    int start = num_per_thread * thread_id;
    int end   = (thread_id == (omp_get_num_threads() - 1)) ? MSG_COUNT : (start + num_per_thread);

    for (int i = start; i < end; ++i) {
      std::vector<float> vec;
      for (int j = 0; j < i; ++j) {
        vec.emplace_back((float)j);
      }
      nontrivial_t nontrivial((uint64_t)i, std::move(vec));
      send(0, nontrivial);
    }
  };

  auto __recv = [&](int p_i, plato::bsp_recv_pmsg_t<nontrivial_t>& pmsg) {
    lock.lock();
    message_map[pmsg->a_] = *pmsg;
    lock.unlock();
  };

  ASSERT_EQ(0, plato::fine_grain_bsp<nontrivial_t>(__send, __recv, opt));
  ASSERT_EQ(MSG_COUNT, message_map.size());

  for (uint64_t i = 0; i < MSG_COUNT; ++i) {
    ASSERT_EQ(1, message_map.count(i));
    ASSERT_EQ(i, message_map[i].b_.size());
    for (uint64_t j = 0; j < i; ++j) {
      ASSERT_EQ(j, message_map[i].b_[j]);
    }
  }
}

TEST(Parallel, OriginBSPTrivial) {
  init_cluster_info();

  plato::bsp_opts_t opt;
  opt.threads_              = 3;
  opt.flying_send_per_node_ = 3;
  opt.flying_recv_          = 3;
  opt.global_size_          = 128;
  opt.local_capacity_       = 3;

  const int MSG_COUNT = 1033;

  plato::spinlock_t lock;
  std::map<int, int> num_count;

  auto bsp_send = [&](plato::bsp_send_callback_t<int> send) {
    int thread_id = omp_get_thread_num();
    int num_per_thread = MSG_COUNT / omp_get_num_threads();
    int start = num_per_thread * thread_id;
    int end   = (thread_id == (omp_get_num_threads() - 1)) ? MSG_COUNT : (start + num_per_thread);

    for (int i = start; i < end; ++i) {
      send(0, i);
    }
  };

  size_t recvd = 0;
  auto bsp_recv = [&](int p_i, plato::bsp_recv_pmsg_t<int>& pmsg) {
    __sync_fetch_and_add(&recvd, 1);

    lock.lock();
    num_count[*pmsg]++;
    lock.unlock();
  };

  plato::bsp_buffer_t recv_buff;
  ASSERT_EQ(0, plato::bsp<int>(&recv_buff, bsp_send, opt));

  plato::bsp_chunk_vistor_t vistor(recv_buff);
  #pragma omp parallel
  {
    while (vistor.next_chunk<int>(bsp_recv)) { }
  }

  for (int i = 0; i < MSG_COUNT; ++i) {
    ASSERT_THAT(num_count, testing::Contains(std::make_pair(i, 1)));
  }
}

TEST(Parallel, OriginBSPNonTrivial) {
  init_cluster_info();

  plato::bsp_opts_t opt;
  opt.threads_              = 3;
  opt.flying_send_per_node_ = 3;
  opt.flying_recv_          = 3;
  opt.global_size_          = 33;
  opt.local_capacity_       = 3;

  const int MSG_COUNT = 1033;

  plato::spinlock_t lock;
  std::map<uint64_t, nontrivial_t> message_map;

  auto __send = [&](plato::bsp_send_callback_t<nontrivial_t> send) {
    int thread_id = omp_get_thread_num();
    int num_per_thread = MSG_COUNT / omp_get_num_threads();
    int start = num_per_thread * thread_id;
    int end   = (thread_id == (omp_get_num_threads() - 1)) ? MSG_COUNT : (start + num_per_thread);

    for (int i = start; i < end; ++i) {
      std::vector<float> vec;
      for (int j = 0; j < i; ++j) {
        vec.emplace_back((float)j);
      }
      nontrivial_t nontrivial((uint64_t)i, std::move(vec));
      send(0, nontrivial);
    }
  };

  auto __recv = [&](int p_i, plato::bsp_recv_pmsg_t<nontrivial_t>& pmsg) {
    lock.lock();
    message_map[pmsg->a_] = *pmsg;
    lock.unlock();
  };

  plato::bsp_buffer_t recv_buff;
  ASSERT_EQ(0, plato::bsp<nontrivial_t>(&recv_buff, __send, opt));

  plato::bsp_chunk_vistor_t vistor(recv_buff);
  #pragma omp parallel
  {
    while (vistor.next_chunk<nontrivial_t>(__recv)) { }
  }

  ASSERT_EQ(MSG_COUNT, message_map.size());

  for (uint64_t i = 0; i < MSG_COUNT; ++i) {
    ASSERT_EQ(1, message_map.count(i));
    ASSERT_EQ(i, message_map[i].b_.size());
    for (uint64_t j = 0; j < i; ++j) {
      ASSERT_EQ(j, message_map[i].b_[j]);
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

