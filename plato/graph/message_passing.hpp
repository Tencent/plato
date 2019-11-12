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

#ifndef __PLATO_GRAPH_MESSAGE_PASSING_HPP__
#define __PLATO_GRAPH_MESSAGE_PASSING_HPP__

#include <cstdint>
#include <cstdlib>

#include <memory>
#include <utility>
#include <functional>
#include <type_traits>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "plato/graph/base.hpp"
#include "plato/util/bitmap.hpp"
#include "plato/parallel/bsp.hpp"
#include "plato/parallel/broadcast.hpp"

namespace plato {

// ******************************************************************************* //
// aggregate message

template <typename MSG>
struct mepa_ag_message_t {
  vid_t v_i_;
  MSG   message_;

  template<typename Ar>
  void serialize(Ar &ar) {
    ar & v_i_ & message_;
  }
};

template <typename MSG>
using mepa_ag_send_callback_t = std::function<void(const mepa_ag_message_t<MSG>&)>;

template <typename MSG>
struct mepa_ag_context_t {
  mepa_ag_send_callback_t<MSG> send;
};

template <typename MSG, typename ADJ_LIST>
using mepa_ag_merge_t = std::function<void(const mepa_ag_context_t<MSG>&, vid_t, const ADJ_LIST&)>;

template <typename MSG, typename R>
using mepa_ag_sink_t  = std::function<R(int, mepa_ag_message_t<MSG>&)>;

/*
 * Traveling from every vertex, use neighbour's state to generate update message,
 * then send it to target vertex.
 *
 * \tparam MSG        message type
 * \tparam GRAPH      graph structure type
 * \tparam R          return value of sink task
 * 
 * \param graph       Graph structure, dcsc, csc, etc...
 * \param merge_task  merge task, run in parallel
 * \param sink_task   message sink task, run in parallel
 * \param opts        bulk synchronous parallel options
 *
 * \return
 *    sum of sink_task return
 **/
template <typename MSG, typename R, typename GRAPH>
R aggregate_message (
    GRAPH& graph,
    mepa_ag_merge_t<MSG, typename GRAPH::adj_unit_list_spec_t> merge_task,
    mepa_ag_sink_t<MSG, R> sink_task,
    bsp_opts_t bsp_opts = bsp_opts_t()) {

  auto& cluster_info = cluster_info_t::get_instance();
  if (bsp_opts.threads_ <= 0) { bsp_opts.threads_ = cluster_info.threads_; }

  std::vector<R> reducer_vec(bsp_opts.threads_, R());
  thread_local R* preducer;

  // some compiler is not that smart can inline graph->partitioner()->get_partition_id()
  auto partitioner = graph.partitioner();

  auto bsp_send = [&](bsp_send_callback_t<mepa_ag_message_t<MSG>> send) {
    auto send_callback = [&](const mepa_ag_message_t<MSG>& message) {
      send(partitioner->get_partition_id(message.v_i_), message);
    };

    mepa_ag_context_t<MSG> context { send_callback };

    auto traversal = [&](vid_t v_i, const typename GRAPH::adj_unit_list_spec_t& adjs) {
      merge_task(context, v_i, adjs);
      return true;
    };

    size_t chunk_size = 256;
    while (graph.next_chunk(traversal, &chunk_size)) { }
  };

  auto bsp_recv = [&](int p_i, bsp_recv_pmsg_t<mepa_ag_message_t<MSG>>& pmsg) {
    *preducer += sink_task(p_i, *pmsg);
  };

  traverse_opts_t trvs_opts; trvs_opts.mode_ = traverse_mode_t::CIRCLE;
  graph.reset_traversal(trvs_opts);

  int rc = fine_grain_bsp<mepa_ag_message_t<MSG>>(bsp_send, bsp_recv, bsp_opts,
    [&](void) {
      preducer = &reducer_vec[omp_get_thread_num()];
    }
  );
  CHECK(0 == rc);

  R reducer = R();
  #pragma omp parallel for reduction(+:reducer)
  for (size_t i = 0; i < reducer_vec.size(); ++i) {
    reducer += reducer_vec[i];
  }

  R global_reducer;
  MPI_Allreduce(&reducer, &global_reducer, 1, get_mpi_data_type<R>(), MPI_SUM, MPI_COMM_WORLD);

  return global_reducer;
}

// ******************************************************************************* //


// ******************************************************************************* //
// spread message

template <typename MSG>
using mepa_sd_send_callback_t = std::function<void(int, const MSG&)>;

template <typename MSG>
struct mepa_sd_context_t {
  mepa_sd_send_callback_t<MSG> send;
};

template <typename MSG, typename R>
using mepa_sd_sink_t = std::function<R(MSG&)>;

namespace {

template <typename F, typename T>
struct rebind_send_task {
  F func; T v;

  template <typename... Args>
  inline auto operator()(Args... args) const -> decltype(func(v, std::forward<Args>(args)...)) {
    func(v, std::forward<Args>(args)...);
  }
};

template <typename F, typename T>
rebind_send_task<F, T>
bind_send_task(F&& func, T&& v) {
  return { std::forward<F>(func), std::forward<T>(v) };
}

}

/*
 * Traveling from active vertex, generate message, then send it to target 'node'.
 *
 * \tparam MSG        message type
 * \tparam ACTIVE     ACTIVE vertex's container, support bitmap, etc...
 * \tparam R          return value of sink task
 * 
 * \param spread_task spread task, run in parallel, a functor looks like
 *                    void(const mepa_sd_context_t, args...)
 *                    generate message to send
 * \param sink_task   message sink task, run in parallel
 * \param actives     active vertices
 * \param opts        bulk synchronous parallel options
 *
 * \return
 *    sum of sink_task return
 **/
template <typename MSG, typename R, typename ACTIVE, typename SPREAD_FUNC>
R spread_message (
    ACTIVE& actives,
    SPREAD_FUNC&& spread_task,
    mepa_sd_sink_t<MSG, R> sink_task,
    bsp_opts_t bsp_opts = bsp_opts_t()) {

  auto& cluster_info = cluster_info_t::get_instance();
  if (bsp_opts.threads_ <= 0) { bsp_opts.threads_ = cluster_info.threads_; }

  std::vector<R> reducer_vec(bsp_opts.threads_, R());
  thread_local R* preducer;

  auto bsp_send = [&](bsp_send_callback_t<MSG> send) {
    auto send_callback = [&](int node, const MSG& message) {
      send(node, message);
    };

    mepa_sd_context_t<MSG> context { send_callback };

    size_t chunk_size = bsp_opts.local_capacity_;
    auto rebind_traversal = bind_send_task(std::forward<SPREAD_FUNC>(spread_task),
        std::forward<mepa_sd_context_t<MSG>>(context));
    while (actives.next_chunk(rebind_traversal, &chunk_size)) { }
  };

  auto bsp_recv = [&](int /* p_i */, bsp_recv_pmsg_t<MSG>& pmsg) {
    *preducer += sink_task(*pmsg);
  };

  actives.reset_traversal();

  int rc = fine_grain_bsp<MSG>(bsp_send, bsp_recv, bsp_opts,
    [&](void) {
      preducer = &reducer_vec[omp_get_thread_num()];
    }
  );
  CHECK(0 == rc);

  R reducer = R();
  #pragma omp parallel for reduction(+:reducer)
  for (size_t i = 0; i < reducer_vec.size(); ++i) {
    reducer += reducer_vec[i];
  }

  R global_reducer;
  MPI_Allreduce(&reducer, &global_reducer, 1, get_mpi_data_type<R>(), MPI_SUM, MPI_COMM_WORLD);

  return global_reducer;
}

// ******************************************************************************* //


// ******************************************************************************* //
// broadcast message

template <typename MSG>
using mepa_bc_send_callback_t = std::function<void(const MSG&)>;

template <typename MSG>
struct mepa_bc_context_t {
  mepa_bc_send_callback_t<MSG> send;
};

template <typename MSG, typename R>
using mepa_bc_sink_t = std::function<R(int, MSG&)>;

/*
 * Traveling from active vertex, generate message, then broadcast it to every node
 *
 * \tparam MSG          message type
 * \tparam R            return value of sink task
 * \tparam ACTIVE       ACTIVE vertex's container, support bitmap, etc...
 * \tparam SPREAD_FUNC  spread task type
 * 
 * \param spread_task spread task, run in parallel, a functor looks like
 *                    void(const mepa_bc_context_t, args...)
 *                    generate message to broadcast
 * \param sink_task   message sink task, run in parallel
 * \param actives     active vertices
 * \param opts        bulk synchronous parallel options
 *
 * \return
 *    sum of sink_task return
 **/
template <typename MSG, typename R, typename ACTIVE, typename SPREAD_FUNC>
R broadcast_message (
    ACTIVE& actives,
    SPREAD_FUNC&& spread_task,
    mepa_bc_sink_t<MSG, R> sink_task,
    bc_opts_t bc_opts = bc_opts_t()) {

  auto& cluster_info = cluster_info_t::get_instance();
  if (bc_opts.threads_ <= 0) { bc_opts.threads_ = cluster_info.threads_; }

  std::vector<R> reducer_vec(bc_opts.threads_, R());
  thread_local R* preducer;

  auto __send = [&](bc_send_callback_t<MSG> send) {
    auto send_callback = [&](const MSG& message) {
      send(message);
    };

    mepa_bc_context_t<MSG> context { send_callback };

    size_t chunk_size = bc_opts.local_capacity_;
    auto rebind_traversal = bind_send_task(std::forward<SPREAD_FUNC>(spread_task),
        std::forward<mepa_bc_context_t<MSG>>(context));
    while (actives.next_chunk(rebind_traversal, &chunk_size)) { }
  };

  auto __recv = [&](int p_i, bc_recv_pmsg_t<MSG>& pmsg) {
    *preducer += sink_task(p_i, *pmsg);
  };

  actives.reset_traversal();

  int rc = broadcast<MSG>(__send, __recv, bc_opts,
    [&](void) {
      preducer = &reducer_vec[omp_get_thread_num()];
    }
  );
  CHECK(0 == rc);

  R reducer = R();
  #pragma omp parallel for reduction(+:reducer)
  for (size_t i = 0; i < reducer_vec.size(); ++i) {
    reducer += reducer_vec[i];
  }

  R global_reducer;
  MPI_Allreduce(&reducer, &global_reducer, 1, get_mpi_data_type<R>(), MPI_SUM, MPI_COMM_WORLD);

  return global_reducer;
}

// ******************************************************************************* //

}

#endif

