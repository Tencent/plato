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

#ifndef __PLATO_PARALLEL_ALLGATHER_STATE_HPP__
#define __PLATO_PARALLEL_ALLGATHER_STATE_HPP__

#include <poll.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>

#include <ctime>
#include <cstdint>
#include <cstdlib>

#include <list>
#include <mutex>
#include <tuple>
#include <atomic>
#include <chrono>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <condition_variable>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "plato/graph/base.hpp"
#include "plato/graph/state/dense_state.hpp"
#include "plato/graph/partition/sequence.hpp"
#include "plato/util/stream.hpp"
#include "plato/util/archive.hpp"
#include "plato/util/spinlock.hpp"
#include "plato/util/concurrentqueue.h"
#include "plato/parallel/mpi.hpp"


namespace plato {

// ******************************************************************************* //
// all-gather all states to all nodes in the cluster

struct allgather_state_opts_t {
  int      threads_         = -1;            // -1 means all available threads
};


/*
 * high level communication abstraction, all-gather vertex states
 * NOTICE: Currently only allow dense_state_t has POD types.
 *
 * \param 
 * \param  state              dense_state_t, graph vertex state.
 * \param  opts               allgather_state options
 *
 * \return  0 -- success, else -- failed
 **/
template <typename T, typename PART_IMPL>
int allgather_state (
  plato::dense_state_t<T, PART_IMPL>& state,
  allgather_state_opts_t opts = allgather_state_opts_t()) {
  LOG(FATAL) << "Currently only support sequence_balanced_by_destination_t partitioner.";
  return -1;
}

template <typename T>
int allgather_state (
  plato::dense_state_t<T, sequence_balanced_by_destination_t>& state,
  allgather_state_opts_t opts = allgather_state_opts_t()) {

  using partition_t = std::shared_ptr<plato::sequence_balanced_by_destination_t>;

  auto& cluster_info = cluster_info_t::get_instance();
  if (opts.threads_ <= 0) { opts.threads_ = cluster_info.threads_; }
  
  partition_t partitioner = state.partitioner();

  int partitions = cluster_info.partitions_;
  int partition_id = cluster_info.partition_id_;

  vid_t v_begin = partitioner->offset_[partition_id];
  vid_t v_end = partitioner->offset_[partition_id+1];

  std::vector<T> sendbuf(v_end - v_begin);
  #pragma omp parallel for num_threads(opts.threads_) schedule(static, 32)
  for (vid_t vtx = v_begin; vtx < v_end; vtx ++) { 
    sendbuf[vtx - v_begin] = state[vtx];
  }
  
  std::vector<int> recvcounts(partitions);
  std::vector<int> displs(partitions);

  for (int i = 0; i < partitions; i ++) {
    recvcounts[i] = partitioner->offset_[i + 1] - partitioner->offset_[i];
    displs[i] = partitioner->offset_[i];
  }

  CHECK(displs[partitions-1] >= 0) << "Allgatherv exceed int32 limit. displs[partitions-1]: " << displs[partitions-1];
  //LOG(INFO) << "AllgatherState " << partition_id << " Begin ";

  int rc = MPI_Allgatherv(sendbuf.data(), (int)sendbuf.size(), 
                          get_mpi_data_type<T>(),
                          &state[0], recvcounts.data(), displs.data(), 
                          get_mpi_data_type<T>(), MPI_COMM_WORLD);
  CHECK_EQ(rc, MPI_SUCCESS) << "allgather_state MPI_Allgatherv failed, code: " 
                            << rc << " partition_id: " << partition_id;
  //LOG(INFO) << "AllgatherState " << partition_id << " Done ";
  MPI_Barrier(MPI_COMM_WORLD);
  return !(rc == MPI_SUCCESS);
}

// ******************************************************************************* //

} // namespace plato

#endif

