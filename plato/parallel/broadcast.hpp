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

#ifndef __PLATO_PARALLEL_BROADCAST_HPP__
#define __PLATO_PARALLEL_BROADCAST_HPP__

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
#include "plato/util/stream.hpp"
#include "plato/util/archive.hpp"
#include "plato/util/spinlock.hpp"
#include "plato/util/concurrentqueue.h"
#include "plato/parallel/mpi.hpp"

namespace plato {

// ******************************************************************************* //
// broadcast-message to all nodes in the cluster then do computation

struct bc_opts_t {
  int      threads_         = -1;            // -1 means all available threads
  int      flying_recv_     = -1;            // -1 means equal to #node in the cluster
  int      flying_send_     = -1;            // -1 means equal to #node in the cluster
  uint32_t global_size_     = 16 * MBYTES;   // at most global_size_ bytes will be cached
                                             // per one request
  int      local_capacity_  = 4 * PAGESIZE;
  uint32_t batch_size_      = 1;             // batch process #batch_size_ messages

  bool     include_self_    = true;          // broadcast messages to self-node or not
};

template <typename MSG>
using bc_send_callback_t = std::function<bool(const MSG&)>;

template <typename MSG>
using bc_send_task_t = std::function<void(bc_send_callback_t<MSG>)>;

// std::unique_ptr<MSG>
template <typename MSG>
using bc_recv_pmsg_t = typename iarchive_t<MSG, mem_istream_t>::pmsg_t;

// recvd-partition-id, send-callback => void
template <typename MSG>
using bc_recv_task_t = std::function<void(int, bc_recv_pmsg_t<MSG>&)>;

namespace broadcast_detail {

struct chunk_tail_t {
  uint32_t count_;
  uint32_t size_;
} __attribute__((packed));

/**
 * @brief
 * @tparam OARCHIVE_T
 * @param poarchive
 * @return
 */
template <typename OARCHIVE_T>
int append_chunk_tail(OARCHIVE_T* poarchive) {
  chunk_tail_t tail {
    (uint32_t)poarchive->count(),
    (uint32_t)poarchive->size() + (uint32_t)sizeof(chunk_tail_t)
  };
  poarchive->get_stream()->write(&tail, sizeof(tail));
  return 0;
}

struct chunk_desc_t {
  char*    data_;
  uint32_t size_;
  uint32_t count_;
  int      index_;
  int      from_;
};

/**
 * @brief void
 */
void dummy_func(void) { }

}

/*
 * high level communication abstraction, broadcast
 *
 * \param  send_task          produce task, run in parallel
 * \param  recv_task          consume task, run in parallel
 * \param  opts               broadcast options
 * \param  before_recv_task   run before recv task, multi-times, one per thread
 * \param  after_recv_task    run after recv task, multi-times, one per thread
 *
 * \return  0 -- success, else -- failed
 **/
template <typename MSG>
int broadcast (
  bc_send_task_t<MSG> send_task,
  bc_recv_task_t<MSG> recv_task,
  bc_opts_t opts = bc_opts_t(),
  std::function<void(void)> before_recv_task = broadcast_detail::dummy_func,
  std::function<void(void)> after_recv_task  = broadcast_detail::dummy_func) {

  using namespace broadcast_detail;
  using oarchive_spec_t = oarchive_t<MSG, mem_ostream_t>;
  using iarchive_spec_t = iarchive_t<MSG, mem_istream_t>;

  auto& cluster_info = cluster_info_t::get_instance();
  if (opts.threads_ <= 0) { opts.threads_ = cluster_info.threads_; }
  if (opts.flying_recv_ <= 0) { opts.flying_recv_ = cluster_info.partitions_; }
  if (opts.flying_send_ <= 0) { opts.flying_send_ = cluster_info.partitions_; }

  std::atomic<bool> process_continue(true);
  // pin all these to numa node ??
  moodycamel::ConcurrentQueue<chunk_desc_t> chunk_queue;
  std::vector<std::shared_ptr<char>>        buffs_vec(opts.flying_recv_);
  std::unique_ptr<std::atomic<size_t>[]>    chunk_left(new std::atomic<size_t>[opts.flying_recv_]);

  const uint64_t buff_size = 2UL * 1024UL * (uint64_t)MBYTES - 1;
  for (size_t r_i = 0; r_i < buffs_vec.size(); ++r_i) {
    char* buff = (char*)mmap(nullptr, buff_size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    buffs_vec[r_i].reset(buff, [](char* p) { munmap(p, buff_size); });
    chunk_left[r_i].store(0);
  }

  std::thread recv_assist_thread([&](void) {
    std::atomic<int>         finished_count(0);
    std::vector<bool>        processing(opts.flying_recv_, false);
    std::vector<MPI_Request> requests_vec(opts.flying_recv_, MPI_REQUEST_NULL);

    for (size_t r_i = 0; r_i < requests_vec.size(); ++r_i) {
      MPI_Irecv(buffs_vec[r_i].get(), buff_size, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &requests_vec[r_i]);
    }

    auto probe_once =
      [&](bool continued) {
        int  flag        = 0;
        int  index       = 0;
        int  recv_bytes  = 0;
        bool has_message = false;
        MPI_Status status;

        MPI_Testany(requests_vec.size(), requests_vec.data(), &index, &flag, &status);
        while (flag && (MPI_UNDEFINED != index)) {
          if (ShuffleFin == status.MPI_TAG) {
            ++finished_count;
          } else {  // call recv_task
            char* buff = buffs_vec[index].get();

            MPI_Get_count(&status, MPI_CHAR, &recv_bytes);
            CHECK((size_t)recv_bytes >= sizeof(chunk_tail_t)) << "recv message too small: " << recv_bytes;

            while (recv_bytes > 0) {  // push task in queue
              chunk_tail_t* tail = (chunk_tail_t*)(&buff[recv_bytes - sizeof(chunk_tail_t)]);
              CHECK(tail->size_ >= sizeof(chunk_tail_t));
              char* data = &buff[recv_bytes] - tail->size_;

              ++chunk_left[index];
              chunk_queue.enqueue(chunk_desc_t { data, tail->size_ - (uint32_t)sizeof(chunk_tail_t),
                                                 tail->count_, index, status.MPI_SOURCE });
              recv_bytes -= (int)tail->size_;
            }
            CHECK(0 == recv_bytes);

          }
          requests_vec[index] = MPI_REQUEST_NULL;
          processing[index] = true;

          has_message = true;
          if (false == continued) { break; }

          MPI_Testany(requests_vec.size(), requests_vec.data(), &index, &flag, &status);
        }

        return has_message;
      };

    auto restart_once = [&](void) {
      bool found = false;
      for (size_t i = 0; i < processing.size(); ++i) {
        if (processing[i] && (0 == chunk_left[i].load())) {
          found = true;
          processing[i] = false;
          MPI_Irecv(buffs_vec[i].get(), buff_size, MPI_CHAR, MPI_ANY_SOURCE,
              MPI_ANY_TAG, MPI_COMM_WORLD, &requests_vec[i]);
        }
      }
      return found;
    };

    uint32_t idle_times = 0;
    while (finished_count < cluster_info.partitions_) {
      bool busy = probe_once(false);
      busy = restart_once() || busy;

      idle_times += (uint32_t)(false == busy);
      if (idle_times > 10) {
        poll(nullptr, 0, 1);
        idle_times = 0;
      } else if (false == busy) {
        pthread_yield();
      }
    }

    // we must probe here when use multi-threads. when one thread start irecv and release cpu for a while
    // another thread may receive finished signal and made finished_count bigger than cluster_info.partitions_,
    // then when first thread wake up, it will not process last received messages.
    probe_once(true);

    bool busy = false;
    do {  // wait all tasks finished
      busy = false;
      for (size_t i = 0; i < processing.size(); ++i) {
        if (processing[i] && (0 != chunk_left[i].load())) {
          busy = true;
        }
      }
      if (busy) { poll(nullptr, 0, 1); }
    } while (busy);

    for (size_t r_i = 0; r_i < requests_vec.size(); ++r_i) {
      if (MPI_REQUEST_NULL != requests_vec[r_i]) {
        MPI_Cancel(&requests_vec[r_i]);
        MPI_Wait(&requests_vec[r_i], MPI_STATUS_IGNORE);
      }
    }

    process_continue.store(false);
  });

  std::atomic<int> cpus(0);

  std::thread recv_thread ([&](void) {
    #pragma omp parallel num_threads(opts.threads_)
    {
      auto yeild = [&](bool inc, bool should_sleep) {
        if (inc) { ++cpus; }

        if (should_sleep) {
          poll(nullptr, 0, 1);
        } else {
          pthread_yield();
        }

        int times = 0;
        while (cpus.fetch_sub(1) <= 0) {
          ++cpus;
          if (++times > 10) {
            poll(nullptr, 0, 1);
            times = 0;
          } else {
            pthread_yield();
          }
        }
      };

      auto probe_once =
        [&](uint32_t batch_size) {
          bool has_message = false;
          chunk_desc_t chunk { nullptr, 0, 0, 0, 0 };

          uint32_t processed = 0;
          while (chunk_queue.try_dequeue(chunk)) {
            iarchive_spec_t iarchive(chunk.data_, chunk.size_, chunk.count_);
            for (auto msg = iarchive.absorb(); nullptr != msg; msg = iarchive.absorb()) {
              recv_task(chunk.from_, msg);
            }
            --chunk_left[chunk.index_];

            has_message = true;
            if (++processed >= batch_size) { break; }
          }

          return has_message;
        };

      uint32_t idles = 0;

      yeild(false, true);
      before_recv_task();
      while (process_continue) {
        idles += (uint32_t)(false == probe_once(opts.batch_size_));

        if (idles > 3) {
          yeild(true, true);
          idles = 0;
        } else {
          yeild(true, false);
        }
      }
      after_recv_task();
    }
  });

  spinlock_t buflck;
  std::list<mem_ostream_t> global_sndbuf_list;

  spinlock_t reqlck;
  std::list<std::pair<std::vector<MPI_Request>, mem_ostream_t>> flying_requests_list;

#ifdef __BROADCAST_DEBUG__
  volatile bool perf_continue = true;
  std::thread perf_thread([&](void) {
    time_t __stump = 0;

    LOG(INFO) << "[PERF] threads_:        " << opts.threads_;
    LOG(INFO) << "[PERF] flying_recv_:    " << opts.flying_recv_;
    LOG(INFO) << "[PERF] flying_send_:    " << opts.flying_send_;
    LOG(INFO) << "[PERF] global_size_:    " << opts.global_size_;
    LOG(INFO) << "[PERF] local_capacity_: " << opts.local_capacity_;
    LOG(INFO) << "[PERF] batch_size_:     " << opts.batch_size_;
    LOG(INFO) << "[PERF] include_self_:   " << opts.include_self_;

    while (perf_continue) {
      if (time(nullptr) - __stump < 10) { poll(nullptr, 0, 1); continue; }
      __stump = time(nullptr);

      LOG(INFO) << "[PERF] cpus: " << cpus.load();

      __asm volatile ("pause" ::: "memory");

      LOG(INFO) << "[PERF] sndbuf: " << global_sndbuf_list.size();
      LOG(INFO) << "[PERF] flying: " << flying_requests_list.size();
    }
  });
#endif

  for (int r_i = 0; r_i < opts.flying_send_; ++r_i) {
    global_sndbuf_list.emplace_back((opts.global_size_ / PAGESIZE + 1) * PAGESIZE);
  }

  std::atomic<bool> continued(true);
  std::thread send_assist_thread ([&](void) {  // move complete request to buffer
    uint32_t idle_times = 0;
    while (continued.load()) {
      bool busy = false;
      std::list<std::pair<std::vector<MPI_Request>*, decltype(flying_requests_list.begin())>>
        all_requests;

      reqlck.lock();
      for (auto it = flying_requests_list.begin(); flying_requests_list.end() != it; ++it) {
        all_requests.emplace_back(std::move(std::make_pair(&it->first, it)));
      }
      reqlck.unlock();

      if (0 == all_requests.size()) { continue; }

      for (auto& request: all_requests) {
        int flag = 0;
        CHECK(MPI_SUCCESS == MPI_Testall(request.first->size(), request.first->data(), &flag, MPI_STATUSES_IGNORE));

        if (flag) {
          auto& it = request.second;
          mem_ostream_t oss(std::move(it->second));

          reqlck.lock();
          flying_requests_list.erase(it);
          reqlck.unlock();

          oss.reset();

          buflck.lock();
          global_sndbuf_list.emplace_back(std::move(oss));
          buflck.unlock();

          busy = true;
        }
      }

      idle_times += (uint32_t)(false == busy);
      if (idle_times > 10) {
        poll(nullptr, 0, 1);
        idle_times = 0;
      } else if (false == busy) {
        pthread_yield();
      }
    }
  });

  #pragma omp parallel num_threads(opts.threads_)
  {
    oarchive_spec_t oarchive(16 * PAGESIZE);

    auto yeild = [&](bool should_sleep) {
      ++cpus;

      if (should_sleep) {
        poll(nullptr, 0, 1);
      } else {
        pthread_yield();
      }

      int times = 0;
      while (cpus.fetch_sub(1) <= 0) {
        ++cpus;

        if (++times > 10) {
          poll(nullptr, 0, 1);
          times = 0;
        } else {
          pthread_yield();
        }
      }
    };

    auto flush_local = [&](void) {
      uint32_t idles = 0;

      buflck.lock();
      while (0 == global_sndbuf_list.size()) {  // wait for available slots
        buflck.unlock();
        if (++idles > 3) {
          yeild(true);
          idles = 0;
        } else {
          yeild(false);
        }
        buflck.lock();
      }

      append_chunk_tail(&oarchive);
      auto chunk_buff = oarchive.get_intrusive_buffer();
      global_sndbuf_list.back().write(chunk_buff.data_, chunk_buff.size_);

      if (global_sndbuf_list.back().size() > opts.global_size_) {  // start a new ISend
        mem_ostream_t oss(std::move(global_sndbuf_list.back()));
        auto buff = oss.get_intrusive_buffer();

        global_sndbuf_list.pop_back();
        buflck.unlock();

        std::vector<MPI_Request> requests;
        if (opts.include_self_) {
          requests.resize(cluster_info.partitions_);
        } else {
          requests.resize(cluster_info.partitions_ - 1);
        }

        if (requests.size() > 0) {
          reqlck.lock();
          flying_requests_list.emplace_back(std::move(std::make_pair(std::move(requests), std::move(oss))));

          {
            int p_i = (cluster_info.partition_id_ + 1) % cluster_info.partitions_;
            for (auto& req: flying_requests_list.back().first) {
              MPI_Isend(buff.data_, buff.size_, MPI_CHAR, p_i, Shuffle, MPI_COMM_WORLD, &req);
              p_i = (p_i + 1) % cluster_info.partitions_;
            }
          }
          reqlck.unlock();
        } else {  // poor man, what's wrong with you...
          LOG(WARNING) << "broadcast to no one...";
          buflck.lock();
          global_sndbuf_list.emplace_back(std::move(oss));
          buflck.unlock();
        }
      } else {
        buflck.unlock();
      }

      oarchive.reset();
    };

    auto send_callback =
      [&](const MSG& msg) {
        oarchive.emit(msg);
        if (oarchive.count() >= (size_t)opts.local_capacity_) {  // flush oarchive, use size() will hurt performance
          flush_local();
        }
        return true;
      };

    send_task(send_callback);

    if (oarchive.count()) { flush_local(); }
    ++cpus;
  }

#ifdef __BROADCAST_DEBUG__
  LOG(INFO) << cluster_info.partition_id_ << " - send finished";
#endif

  continued.store(false);
  send_assist_thread.join();

  {  // flush global
    for (auto& oss: global_sndbuf_list) {
      if (oss.size()) {
        auto buff = oss.get_intrusive_buffer();

        std::vector<MPI_Request> requests;
        if (opts.include_self_) {
          requests.resize(cluster_info.partitions_);
        } else {
          requests.resize(cluster_info.partitions_ - 1);
        }

        if (requests.size() > 0) {
          flying_requests_list.emplace_back(std::move(std::make_pair(std::move(requests), std::move(oss))));
          {
            int p_i = (cluster_info.partition_id_ + 1) % cluster_info.partitions_;
            for (auto& req: flying_requests_list.back().first) {
              MPI_Isend(buff.data_, buff.size_, MPI_CHAR, p_i, Shuffle, MPI_COMM_WORLD, &req);
              p_i = (p_i + 1) % cluster_info.partitions_;
            }
          }
        }
      }
    }

    for (auto& request: flying_requests_list) {
      CHECK(MPI_SUCCESS == MPI_Waitall(request.first.size(), request.first.data(), MPI_STATUSES_IGNORE));
    }
  }

  for (int p_i = 0; p_i < cluster_info.partitions_; ++p_i) {  //  broadcast finish signal
    MPI_Send(nullptr, 0, MPI_CHAR, p_i, ShuffleFin, MPI_COMM_WORLD);
  }

  recv_assist_thread.join();
  recv_thread.join();
  MPI_Barrier(MPI_COMM_WORLD);

#ifdef __BROADCAST_DEBUG__
  perf_continue = false;
  perf_thread.join();
#endif

  return 0;
}

// ******************************************************************************* //

}

#endif

