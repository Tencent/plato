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

#include <cstdint>
#include <cstdlib>

#include <atomic>
#include <thread>
#include <future>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "plato/graph/graph.hpp"

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

const size_t kMaxIdx = 64000000;

size_t one_shot_bsp(MPI_Comm comm) {
  using send_callback_t = plato::bsp_send_callback_t<size_t>;
  using recv_msp_t      = plato::bsp_recv_pmsg_t<size_t>;

  auto& cluster_info = plato::cluster_info_t::get_instance();

  std::atomic<size_t> send_idx(0);
  std::atomic<size_t> recv_idx(0);

  plato::bsp_opts_t opts;
  opts.comm_    = comm;
  opts.threads_ = FLAGS_threads / 2;

  int peer = cluster_info.partition_id_ == 0 ? 1 : 0;

  auto __send = [&](send_callback_t send) {
    size_t idx = 0;
    while ((idx = send_idx.fetch_add(64)) < kMaxIdx) {
      for (int i = 0; i < 64; ++i) {
        send(peer, 1);
      }
    }
  };

  auto __recv = [&](int, recv_msp_t& pmsg) {
    recv_idx += *pmsg;
  };

  plato::fine_grain_bsp<size_t>(__send, __recv, opts);
  return recv_idx.load();
}

int main(int argc, char** argv) {
  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  MPI_Comm comm1, comm2;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm1);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm2);

  auto fut1 = std::async(std::launch::async, one_shot_bsp, comm1);
  auto fut2 = std::async(std::launch::async, one_shot_bsp, comm2);

  LOG(INFO) << fut1.get() <<  "," << fut2.get();

  return 0;
}

