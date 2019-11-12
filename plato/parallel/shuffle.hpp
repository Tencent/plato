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

#ifndef __PLATO_PARALLEL_SHUFFLE_HPP__
#define __PLATO_PARALLEL_SHUFFLE_HPP__

#include <poll.h>
#include <unistd.h>
#include <sys/mman.h>

#include <cstdint>
#include <cstdlib>

#include <list>
#include <tuple>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "plato/graph/base.hpp"
#include "plato/util/stream.hpp"
#include "plato/util/archive.hpp"
#include "plato/parallel/mpi.hpp"

namespace plato {

/***************************************************************************/
// structs for shuffle

// partition-id, message => send successfully or not
template <typename MSG>
using shuffle_send_callback_t = std::function<bool(int, const MSG&)>;

// send-callback => void
template <typename MSG>
using shuffle_send_task_t = std::function<void(shuffle_send_callback_t<MSG>)>;

// std::unique_ptr<MSG>
template <typename MSG>
using shuffle_recv_pmsg_t = typename iarchive_t<MSG, mem_istream_t>::pmsg_t;

// partition-id, send-callback => void
template <typename MSG>
using shuffle_recv_task_t = std::function<void(int, shuffle_recv_pmsg_t<MSG>&)>;

struct shuffle_opts_t {
  int    send_threads_             = -1;
  int    recv_threads_             = -1;
  int    flying_send_req_per_node_ = 3;
  int    flying_recv_req_          = 3;
  size_t obuf_size_                = HUGESIZE;
  size_t ibuf_size_                = HUGESIZE;
  size_t obj_per_one_msg_          = PAGESIZE;
};

namespace {  // helper functions

struct shuffle_tail_t {
  uint32_t count_;
} __attribute__((packed));

template <typename OARCHIVE_T>
int append_shuffle_tail(OARCHIVE_T* poarchive) {  // append info at the end of message
  shuffle_tail_t tail { (uint32_t)poarchive->count() };
  poarchive->get_stream()->write(&tail, sizeof(tail));
  return 0;
}

}  // namespace anonymous

/*
 * high level communication abstraction, shuffle
 *
 * \param  send_task     send task callback
 * \param  recv_task     receive task callback
 * \param  sf_ops        shuffle options
 *
 * \return  0 -- success, else -- failed
 **/
template <typename MSG_T>
inline int shuffle (
  shuffle_send_task_t<MSG_T> send_task,
  shuffle_recv_task_t<MSG_T> recv_task,
  shuffle_opts_t sf_ops = shuffle_opts_t()) {

  using oarchive_spec_t = oarchive_t<MSG_T, mem_ostream_t>;
  using iarchive_spec_t = iarchive_t<MSG_T, mem_istream_t>;

  auto& cluster_info = cluster_info_t::get_instance();

  if (-1 == sf_ops.send_threads_) { sf_ops.send_threads_ = std::max(1, cluster_info.threads_ / 2); }
  if (-1 == sf_ops.recv_threads_) { sf_ops.recv_threads_ = std::max(1, cluster_info.threads_ / 2); }

  //  recv tasks
  std::thread recv_thread([&](void) {
    volatile int finished_count = 0;

    #pragma omp parallel num_threads(sf_ops.recv_threads_)
    {
      const uint64_t buff_size = 2UL * 1024UL * (uint64_t)MBYTES - 1;

      std::vector<MPI_Request> requests_vec(sf_ops.flying_recv_req_, MPI_REQUEST_NULL);
      std::vector<std::shared_ptr<char>> buffs_vec(sf_ops.flying_recv_req_);

      // start async receive
      for (size_t r_i = 0; r_i < requests_vec.size(); ++r_i) {
        char* buff = (char*)mmap(nullptr, buff_size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        buffs_vec[r_i].reset(buff, [](char* p) { munmap(p, buff_size); });

        MPI_Irecv(buff, buff_size, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &requests_vec[r_i]);
      }

      auto probe_once = [&](void) {
        int flag       = 0;
        int index      = 0;
        int recv_bytes = 0;
        MPI_Status status;

        MPI_Testany(requests_vec.size(), requests_vec.data(), &index, &flag, &status);
        while (flag) {
          if (ShuffleFin == status.MPI_TAG) {
            __sync_fetch_and_add(&finished_count, 1);
          } else {  // call recv_task
            char* buff = buffs_vec[index].get();

            MPI_Get_count(&status, MPI_CHAR, &recv_bytes);
            CHECK(recv_bytes >= static_cast<int>(sizeof(shuffle_tail_t))) << "recv message too small: " << recv_bytes;

            shuffle_tail_t* tail = (shuffle_tail_t*)(&buff[recv_bytes - sizeof(shuffle_tail_t)]);
            iarchive_spec_t iarchive(buff, recv_bytes - sizeof(shuffle_tail_t), tail->count_);

            for (auto msg = iarchive.absorb(); nullptr != msg; msg = iarchive.absorb()) {
              recv_task(status.MPI_SOURCE, msg);
            }
          }

          // start a new irecv
          MPI_Irecv(buffs_vec[index].get(), buff_size, MPI_CHAR, MPI_ANY_SOURCE,
              MPI_ANY_TAG, MPI_COMM_WORLD, &requests_vec[index]);

          MPI_Testany(requests_vec.size(), requests_vec.data(), &index, &flag, &status);
        }
      };

      while (finished_count < cluster_info.partitions_) {
        probe_once();
        // poll(nullptr, 0, 1);
        pthread_yield();
      }

      // we must probe here when use multi-threads. when one thread start irecv and release cpu for a while
      // another thread may receive finished signal and made finished_count bigger than cluster_info.partitions_,
      // then when first thread wake up, it will not process last received messages.
      probe_once();

      for (size_t r_i = 0; r_i < requests_vec.size(); ++r_i) {
        MPI_Cancel(&requests_vec[r_i]);
        MPI_Wait(&requests_vec[r_i], MPI_STATUS_IGNORE);
      }
    }
  });

  //  send tasks
  #pragma omp parallel num_threads(sf_ops.send_threads_)
  {
    std::vector<std::list<std::shared_ptr<oarchive_spec_t>>> oarchives_vec(cluster_info.partitions_);
    for (size_t p_i = 0; p_i < oarchives_vec.size(); ++p_i) {
      for (int r_i = 0; r_i < sf_ops.flying_send_req_per_node_; ++r_i) {
        oarchives_vec[p_i].emplace_back(new oarchive_spec_t(sf_ops.obuf_size_));
      }
    }

    std::vector<std::list<std::pair<MPI_Request, std::shared_ptr<oarchive_spec_t>>>> flying_requests_vec(cluster_info.partitions_);

    auto send_callback = [&](int p_i, const MSG_T& msg) {
      CHECK((p_i < cluster_info.partitions_) && (p_i >= 0));

      if (static_cast<size_t >(sf_ops.flying_send_req_per_node_) == flying_requests_vec[p_i].size()) {  // waiting for an available oarchive
        auto& requests = flying_requests_vec[p_i];
        int  flag  = 0;
        bool found = false;

        while (true) {
          auto r_it = requests.begin();
          while (requests.end() != r_it) {
            MPI_Test(&r_it->first, &flag, MPI_STATUS_IGNORE);
            if (flag) {
              r_it->second->reset();
              oarchives_vec[p_i].emplace_back(std::move(r_it->second));
              r_it = requests.erase(r_it);
              found = true;
            } else {
              ++r_it;
            }
          }
          if (found) { break; }

          // poll(nullptr, 0, 1);
          pthread_yield();
        }
      }

      auto& poarchive = oarchives_vec[p_i].back();
      poarchive->emit(msg);

      if (poarchive->count() >= sf_ops.obj_per_one_msg_) {  // flush oarchive, use size() will hurt performance
        append_shuffle_tail(poarchive.get());
        auto buff = poarchive->get_intrusive_buffer();

        flying_requests_vec[p_i].emplace_back(std::make_pair(MPI_Request(), poarchive));
        oarchives_vec[p_i].pop_back();

        MPI_Isend(buff.data_, buff.size_, MPI_CHAR, p_i, Shuffle, MPI_COMM_WORLD,
            &flying_requests_vec[p_i].back().first);
      }

      return true;
    };

    send_task(send_callback);

    for (auto& requests: flying_requests_vec) {  // wait all flying requests be done
      if (0 == requests.size()) { continue; }

      std::vector<MPI_Request> requests_vec;
      for (auto& req: requests) { requests_vec.emplace_back(req.first); }

      CHECK(MPI_SUCCESS == MPI_Waitall(requests_vec.size(), requests_vec.data(), MPI_STATUSES_IGNORE));
      requests.clear();
    }

    {  // flush buffers
      for (size_t p_i = 0; p_i < oarchives_vec.size(); ++p_i) {
        auto& oarchives = oarchives_vec[p_i];
        auto& requests  = flying_requests_vec[p_i];

        for (auto& poarchive: oarchives) {
          if (poarchive->size()) {
            append_shuffle_tail(poarchive.get());
            auto buff = poarchive->get_intrusive_buffer();

            requests.emplace_back(std::make_pair(MPI_Request(), poarchive));
            MPI_Isend(buff.data_, buff.size_, MPI_CHAR, p_i, Shuffle, MPI_COMM_WORLD,
                &requests.back().first);
          }
        }
      }

      for (auto& requests: flying_requests_vec) {  // wait all flying requests be done
        if (0 == requests.size()) { continue; }
        std::vector<MPI_Request> requests_vec;
        for (auto& req: requests) { requests_vec.emplace_back(req.first); }

        CHECK(MPI_SUCCESS == MPI_Waitall(requests_vec.size(), requests_vec.data(), MPI_STATUSES_IGNORE));
        requests.clear();
      }
    }
  }

  for (int p_i = 0; p_i < cluster_info.partitions_; ++p_i) {  //  broadcast finish signal
    MPI_Send(nullptr, 0, MPI_CHAR, p_i, ShuffleFin, MPI_COMM_WORLD);
  }

  recv_thread.join();
  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}

/***************************************************************************/

}

#endif

