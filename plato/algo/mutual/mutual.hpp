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

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "boost/align.hpp"
#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/hdfs.hpp"
#include "plato/graph/base.hpp"
#include "plato/graph/state.hpp"
#include "plato/graph/structure.hpp"
#include "plato/graph/message_passing.hpp"
#include "plato/util/intersection.hpp"
#include "plato/util/to_string.hpp"

#include "plato/util/thread_local_object.h"

namespace plato {

namespace mutual_detail {

extern std::unique_ptr<plato::thread_local_buffer> msg_buffer;
extern std::unique_ptr<plato::thread_local_buffer> dsts_buffer;

template <typename T>
struct mutual_msg_t {
  plato::vid_t src_ = 0;
  plato::vid_t dsts_size_ = 0;
  plato::vid_t* dsts_ = nullptr;
  plato::vid_t msg_size_ = 0;
  T* msg_ = nullptr;

  template<typename Ar>
  void serialize(Ar &ar) {
    if(!dsts_) dsts_ = (plato::vid_t*)dsts_buffer->local();
    if(!msg_) msg_ = (T*)msg_buffer->local();

    ar & src_;
    ar & dsts_size_;
    for (unsigned i = 0; i < dsts_size_; i++) {
      ar & dsts_[i];
    }
    ar & msg_size_;
    for (unsigned i = 0; i < msg_size_; i++) {
      ar & msg_[i];
    }
  }
};

}

struct mutual_opts_t {
  bool need_sort_ = true;
};

/*
 * run mutual on a tcsr graph
 *
 * \tparam T               id type
 * \tparam TCSR            graph type
 * \tparam NEIS_FUNC       neighbours extractor, should implement,
 *                         std::vector<T>& operator() (TCSR::graph_data_t&)
 * \tparam ResultCallback  deal with mutual result, should implement
 *                         void operator() (plato::vid_t src, plato::vid_t dst, T* out, size_t size_out)
 *
 * \param tcsr             tcsr graph
 * \param extract_neis     extract neighbours from graph_data_t
 * \param result_callback  result dealer
 *
 * \return
 *      sum of mutual result size
 **/
template <typename T, typename TCSR, typename NEIS_FUNC, typename ResultCallback>
size_t mutual(
  TCSR& tcsr,
  NEIS_FUNC&& extract_neis,
  ResultCallback&& result_callback,
  plato::bsp_opts_t bsp_opts = plato::bsp_opts_t(),
  mutual_opts_t mutual_opts = mutual_opts_t{}) {

  using namespace mutual_detail;
  using EDATA = typename TCSR::edata_t;

  static_assert(std::is_integral<T>::value, "cmp type must be integer.");

  auto& cluster_info = plato::cluster_info_t::get_instance();
  using adj_unit_list_spec_t = plato::adj_unit_list_t<EDATA>;

  auto& partitioner = *tcsr.partitioner();
  if (bsp_opts.threads_ < 0) bsp_opts.threads_ = cluster_info.threads_;

  plato::stop_watch_t watch;
  watch.mark("t0");

  if (mutual_opts.need_sort_) {
    watch.mark("t1");
    tcsr.reset_traversal();
    size_t chunk_size = 64;

    #pragma omp parallel num_threads(bsp_opts.threads_)
    {
      auto traversal = [&tcsr, &extract_neis] (plato::vid_t v_i, const adj_unit_list_spec_t&) {
        T *begin, *end;
        extract_neis(tcsr[v_i], &begin, &end);
        std::sort(begin, end);
        return true;
      };
      while (tcsr.next_chunk(traversal, &chunk_size)) { }
    }

    LOG_IF(INFO, 0 == cluster_info.partition_id_) << "sort tcsr mutual costs: " << watch.show("t1") / 1000.0 << "s";
  }

  watch.mark("t1");

  // more load balanced than previous work
  std::vector<bool> partitions_mask(cluster_info.partitions_, false);
  std::vector<bool> partitions_interchange_mask(cluster_info.partitions_, false);

  for (int i = 0; i < (cluster_info.partitions_ - 1) / 2; i++) {
    int id = (cluster_info.partition_id_ + i + 1) % cluster_info.partitions_;
    partitions_mask[id] = true;
  }

  partitions_interchange_mask[cluster_info.partition_id_] = true;
  if (cluster_info.partitions_ % 2 == 0) {
    partitions_interchange_mask[(cluster_info.partition_id_ + cluster_info.partitions_ / 2) % cluster_info.partitions_] = true;
  }

  /******************* distributed begin *******************/
  thread_local_buffer out_buffer;
  msg_buffer.reset(new thread_local_buffer());
  dsts_buffer.reset(new thread_local_buffer());
  auto msg_buffer_del_defer = defer([] { msg_buffer.reset(); });
  auto dsts_buffer_del_defer = defer([] { dsts_buffer.reset(); });
  thread_local_counter mutual_counter;
  /******************* distributed end *******************/

  tcsr.reset_traversal();

  auto mutual_intersect = [&] (mutual_msg_t<T>& msg) {
    auto& local_counter = mutual_counter.local();
    for (unsigned i = 0; i < msg.dsts_size_; ++i) {

      plato::vid_t dst = msg.dsts_[i];
      auto& v = tcsr[dst];
      T *begin, *end;
      extract_neis(v, &begin, &end);
      const T* set_a  = begin;
      size_t   size_a = end - begin;

      const T* set_b  = msg.msg_;
      size_t   size_b = msg.msg_size_;

      T* out = (T*)out_buffer.local();
      CHECK(size_a && size_b);
      size_t size_out = intersect(set_a, size_a, set_b, size_b, out);
      CHECK(size_out <= std::min(size_a, size_b))
      << boost::format("size_out: %lu, size_a: %lu, size_b: %lu. ") % size_out % size_a % size_b
      << std::vector<T>(msg.msg_, msg.msg_ + msg.msg_size_) << std::vector<T>(begin, end);

      result_callback(msg.src_, dst, out, size_out);
      local_counter += size_out;
    }
  };

  auto __send = [&] (bsp_send_callback_t<mutual_msg_t<T>> send) {
    mutual_msg_t<T> partition_msg[cluster_info.partitions_];

    auto traversal = [&](vid_t v_i, const adj_unit_list_spec_t& /* adjs */) {
      auto& v_src = tcsr[v_i];

      T *msg_begin, *msg_end;
      extract_neis(v_src, &msg_begin, &msg_end);

      for (int partition_id = 0; partition_id < cluster_info.partitions_; ++partition_id) {
        mutual_msg_t<T>& msg = partition_msg[partition_id];
        msg.src_ = v_i;
        msg.dsts_ = (plato::vid_t*)dsts_buffer->local() + (partition_id * v_src.adjs_size_);
        msg.dsts_size_ = 0;
        msg.msg_ = msg_begin;
        msg.msg_size_ = msg_end - msg_begin;
      }

      for (unsigned i = 0; i < v_src.adjs_size_; ++i) {
        auto& adj = v_src.adjs_[i];
        auto dst = adj.neighbour_;
        auto partition_id = partitioner.get_partition_id(dst);
        bool odd = (v_i + dst) % 2;
        if (partitions_mask[partition_id] || (partitions_interchange_mask[partition_id] && ((odd && v_i < dst) || (!odd && v_i > dst)))) {
          mutual_msg_t<T>& msg = partition_msg[partition_id];
          msg.dsts_[msg.dsts_size_++] = dst;
        }
      }

      for (int partition_id = 0; partition_id < cluster_info.partitions_; ++partition_id) {
        mutual_msg_t<T>& msg = partition_msg[partition_id];
        if (msg.dsts_size_) {
          if (partition_id == cluster_info.partition_id_) {
            mutual_intersect(msg);
          } else {
            send(partition_id, msg);
          }
        }
      }
      return true;
    };

    size_t chunk_size = 1;
    while (tcsr.next_chunk(traversal, &chunk_size)) { }
  };

  auto __recv = [&] (int /* p_i */, bsp_recv_pmsg_t<mutual_msg_t<T>>& pmsg) {
    mutual_intersect(*pmsg);
  };

  auto rc = fine_grain_bsp<mutual_msg_t<T>>(__send, __recv, bsp_opts);
  CHECK(0 == rc) << "bsp failed with code: " << rc;

  size_t mutual_counter_result = mutual_counter.reduce_sum();

  MPI_Allreduce(MPI_IN_PLACE, &mutual_counter_result, 1, get_mpi_data_type<size_t >(), MPI_SUM, MPI_COMM_WORLD);
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "compute/save mutual costs: " << watch.show("t1") / 1000.0 << "s mutual_counter:" << mutual_counter_result;
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "mutual costs: " << watch.show("t0") / 1000.0 << "s";

  return mutual_counter_result;
}

}  // namespace plato

