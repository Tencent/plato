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

#ifndef __PLATO_ENGINE_DUALMODE_HPP__
#define __PLATO_ENGINE_DUALMODE_HPP__

#include <cstdint>
#include <cstdlib>
#include <functional>

#include <string>
#include <memory>
#include <type_traits>

#include "mpi.h"
#include "omp.h"
#include "glog/logging.h"

#include "plato/graph/graph.hpp"

namespace plato {

struct dualmode_engine_opts_t {
  double push_threshold_ = 0.05;  //  active_edges < 1/20, choose push-mode, else pull from all nodes
};

namespace dualmode_detail {

template <typename INCOMING, typename OUTGOING, typename P_O = void, typename P_I = void>
struct partition_traits { };

// I don't know why I have to call std::enable_if to enable SFINAE
// use 'typename INCOMING::partition_t' doesn't work
template <typename INCOMING, typename OUTGOING>
struct partition_traits<INCOMING, OUTGOING,
    typename std::enable_if<std::is_same<nullptr_t, OUTGOING>::value>::type,
    typename std::enable_if<std::is_class<typename INCOMING::partition_t>::value>::type> {
  using type = typename INCOMING::partition_t;
};

template <typename INCOMING, typename OUTGOING>
struct partition_traits<INCOMING, OUTGOING,
    typename std::enable_if<std::is_class<typename OUTGOING::partition_t>::value>::type,
    typename std::enable_if<std::is_same<nullptr_t, INCOMING>::value>::type> {
  using type = typename OUTGOING::partition_t;
};

template <typename INCOMING, typename OUTGOING>
struct partition_traits<INCOMING, OUTGOING,
    typename std::enable_if<std::is_class<typename OUTGOING::partition_t>::value>::type,
    typename std::enable_if<std::is_class<typename INCOMING::partition_t>::value>::type> {
  using type = typename INCOMING::partition_t;
};

}  // namespace dualmode_detail

template <typename INCOMING, typename OUTGOING>
class dualmode_engine_t {
public:

  using incoming_graph_t = INCOMING;
  using outgoing_graph_t = OUTGOING;

  // TODO we'd better seperate vertex's partition from edge partition
  using partition_t = typename dualmode_detail::partition_traits<INCOMING, OUTGOING>::type;

  template <typename T>
  using v_state_t  = dense_state_t<T, partition_t>;

  using v_subset_t = bitmap_t<>;

  /*
   * create dualmode-engine from existed edges, better be used with create_dualmode_seq_from_path
   **/
  dualmode_engine_t(
      std::shared_ptr<INCOMING> in_edges,
      std::shared_ptr<OUTGOING> out_edges,
      const graph_info_t& graph_info,
      const dualmode_engine_opts_t opts = dualmode_engine_opts_t());

  dualmode_engine_t(dualmode_engine_t&&);
  dualmode_engine_t& operator=(dualmode_engine_t&&);

  dualmode_engine_t(const dualmode_engine_t&) = delete;
  dualmode_engine_t& operator=(const dualmode_engine_t&) = delete;

  template <typename T>
  v_state_t<T> alloc_v_state(void);

  v_subset_t alloc_v_subset(void);

  // transpose the graph
  // INCOMING and OUTGOING must have same type, and support std::swap
  // std::enable_if<std::is_same<INCOMING, OUTGOING>::value, void>
  //   transpose(void);

  /*
   * foreach edges in the graph, engine will switch between push-pull mode based on the
   * ACTIVE edges in the graph.
   *
   * \tparam MSG           message type
   * \tparam R             return type of slot
   * \tparam PUSH_SIGNAL   void(const mepa_bc_context_t<MSG>&, vid_t)
   * \tparam PUSH_SLOT     R(int, MSG&) // first int param is node-id not vertex-id !!!
   * \tparam PULL_SIGNAL   void(const mepa_ag_context_t<MSG>&, const dcsc_spec_t::adj_unit_list_spec_t&)
   * \tparam PULL_SLOT     R(int, mepa_ag_message_t<MSG>&) // partition-id, message
   *
   * \param push_signal    push-signal functor, can be nullptr
   * \param push_slot      push-slot functor, can be nullptr
   * \param pull_signal    pull-signal functor, can be nullptr
   * \param pull_slot      pull-slot functor, can be nullptr
   * \param actives        specify ACTIVE vertices in the graph
   *
   * \return
   *          sum of slots' return value
   **/
  template <typename MSG, typename R, typename PUSH_SIGNAL, typename PUSH_SLOT,
      typename PULL_SIGNAL, typename PULL_SLOT>
  R foreach_edges(PUSH_SIGNAL&& push_signal, PUSH_SLOT&& push_slot,
      PULL_SIGNAL&& pull_signal, PULL_SLOT&& pull_slot, v_subset_t& actives);

  // push only
  template <typename MSG, typename R, typename PUSH_SIGNAL, typename PUSH_SLOT>
  R foreach_edges(PUSH_SIGNAL&& signal, PUSH_SLOT&& slot, v_subset_t& actives);

  // pull only
  template <typename MSG, typename R, typename PULL_SIGNAL, typename PULL_SLOT>
  R foreach_edges(PULL_SIGNAL&& signal, PULL_SLOT&& slot);

  std::shared_ptr<INCOMING> in_edges(void)  { return in_edges_;  }
  std::shared_ptr<OUTGOING> out_edges(void) { return out_edges_; }

  // reverse the graph
  void reverse(void)     { reversed_ = !reversed_; }
  bool is_reversed(void) { return reversed_; }

  // ******************************************************************************* //

protected:
  std::shared_ptr<INCOMING>          in_edges_;
  std::shared_ptr<OUTGOING>          out_edges_;
  bool                               reversed_;

  graph_info_t                       graph_info_;
  dualmode_engine_opts_t             opts_;

  std::unique_ptr<v_state_t<vid_t>>  out_degrees_;
};

// ******************************************************************************* //
// implementations

namespace dualmode_detail {

template <typename MSG, typename R, typename GRAPH, typename PULL_SIGNAL, typename PULL_SLOT>
typename std::enable_if<!std::is_same<GRAPH, nullptr_t>::value, R>::type
__foreach_edges(PULL_SIGNAL&& signal, PULL_SLOT&& slot, GRAPH& graph) {
  return aggregate_message<MSG, R, GRAPH>(graph, std::forward<PULL_SIGNAL>(signal), std::forward<PULL_SLOT>(slot));
}

template <typename MSG, typename R, typename GRAPH, typename PULL_SIGNAL, typename PULL_SLOT>
typename std::enable_if<std::is_same<GRAPH, nullptr_t>::value, R>::type
__foreach_edges(PULL_SIGNAL&&, PULL_SLOT&&, GRAPH&) {
  CHECK("do pull-mode without incoming edges");
  return R();
}

template <typename MSG, typename R, typename GRAPH, typename PUSH_SIGNAL, typename PUSH_SLOT>
typename std::enable_if<!std::is_same<GRAPH, nullptr_t>::value, R>::type
__foreach_edges(PUSH_SIGNAL&& signal, PUSH_SLOT&& slot, GRAPH& graph, bitmap_t<>& actives) {
  auto active_view = create_active_v_view(graph.partitioner()->self_v_view(), actives);
  return broadcast_message<MSG, R>(active_view, std::forward<PUSH_SIGNAL>(signal), std::forward<PUSH_SLOT>(slot));
}

template <typename MSG, typename R, typename GRAPH, typename PUSH_SIGNAL, typename PUSH_SLOT>
typename std::enable_if<std::is_same<GRAPH, nullptr_t>::value, R>::type
__foreach_edges(PUSH_SIGNAL&&, PUSH_SLOT&&, GRAPH&, bitmap_t<>&) {
  CHECK("do push-mode without outgoing edges");
  return R();
}

}  // namespace dualmode_detail

template <typename INCOMING, typename OUTGOING>
dualmode_engine_t<INCOMING, OUTGOING>::dualmode_engine_t (
    std::shared_ptr<INCOMING> in_edges,
    std::shared_ptr<OUTGOING> out_edges,
    const graph_info_t& graph_info, const dualmode_engine_opts_t opts)
  : in_edges_(in_edges), out_edges_(out_edges), reversed_(false),
    graph_info_(graph_info), opts_(opts), out_degrees_(nullptr) {
  if (0 == plato::cluster_info_t::get_instance().partition_id_) {
    LOG(INFO) << "create dualmode-engine, push-threshold: " << opts_.push_threshold_;
  }
}

template <typename INCOMING, typename OUTGOING>
dualmode_engine_t<INCOMING, OUTGOING>&
dualmode_engine_t<INCOMING, OUTGOING>::operator=(dualmode_engine_t<INCOMING, OUTGOING>&& x) {
  in_edges_    = std::move(x.in_edges_);
  out_edges_   = std::move(x.out_edges_);
  reversed_    = x.reversed_;
  graph_info_  = x.graph_info_;
  opts_        = x.opts_;
  out_degrees_ = std::move(x.out_degrees_);
  return *this;
}

template <typename INCOMING, typename OUTGOING>
dualmode_engine_t<INCOMING, OUTGOING>::dualmode_engine_t(dualmode_engine_t<INCOMING, OUTGOING>&& x) {
  this->operator=(std::forward<dualmode_engine_t>(x));
}

template <typename INCOMING, typename OUTGOING>
template <typename T>
typename dualmode_engine_t<INCOMING, OUTGOING>::template v_state_t<T>
dualmode_engine_t<INCOMING, OUTGOING>::alloc_v_state(void) {
  return v_state_t<T>(graph_info_.max_v_i_, in_edges_->partitioner());
}

template <typename INCOMING, typename OUTGOING>
typename dualmode_engine_t<INCOMING, OUTGOING>::v_subset_t
dualmode_engine_t<INCOMING, OUTGOING>::alloc_v_subset(void) {
  return v_subset_t(graph_info_.max_v_i_ + 1);
}

template <typename INCOMING, typename OUTGOING>
template <typename MSG, typename R, typename PUSH_SIGNAL, typename PUSH_SLOT,
    typename PULL_SIGNAL, typename PULL_SLOT>
R dualmode_engine_t<INCOMING, OUTGOING>::foreach_edges(PUSH_SIGNAL&& push_signal, PUSH_SLOT&& push_slot,
    PULL_SIGNAL&& pull_signal, PULL_SLOT&& pull_slot, dualmode_engine_t<INCOMING, OUTGOING>::v_subset_t& actives) {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  if (nullptr == out_degrees_) {
    plato::stop_watch_t watch;
    watch.mark("t1");
    out_degrees_.reset(new v_state_t<vid_t>(
      std::move(generate_dense_out_degrees_fg<plato::vid_t>(graph_info_, *in_edges_, false))
    ));
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "generate out-degrees from graph cost: " << watch.show("t1") / 1000.0 << "s";
    }
  }

  eid_t edges = graph_info_.edges_;
  if (false == graph_info_.is_directed_) { edges = edges * 2; }

  eid_t active_edges = 0;

  out_degrees_->reset_traversal(std::shared_ptr<v_subset_t>(&actives, [](v_subset_t*) { }));
  #pragma omp parallel reduction(+:active_edges)
  {
    size_t chunk_size = 4 * PAGESIZE;
    eid_t __active_edges = 0;

    while (out_degrees_->next_chunk([&](vid_t v_i, vid_t* degrees) {
      __active_edges += (*degrees); return true;
    }, &chunk_size)) { }
    active_edges += __active_edges;
  }
  MPI_Allreduce(MPI_IN_PLACE, &active_edges, 1, get_mpi_data_type<eid_t>(), MPI_SUM, MPI_COMM_WORLD);

  bool is_sparse = ((double)active_edges / (double)edges) < opts_.push_threshold_;
  if (is_sparse) {
#ifdef __DUALMODE_DEBUG__
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "active_edges: " << active_edges << "/" << edges << "("
        << (double)active_edges / (double)edges << "), push-mode";
    }
#endif
    return foreach_edges<MSG, R>(std::forward<PUSH_SIGNAL>(push_signal), std::forward<PUSH_SLOT>(push_slot), actives);
  } else {
#ifdef __DUALMODE_DEBUG__
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "active_edges: " << active_edges << "/" << edges << "("
        << (double)active_edges / (double)edges << "), pull-mode";
    }
#endif
    return foreach_edges<MSG, R>(std::forward<PULL_SIGNAL>(pull_signal), std::forward<PULL_SLOT>(pull_slot));
  }
}

template <typename INCOMING, typename OUTGOING>
template <typename MSG, typename R, typename PULL_SIGNAL, typename PULL_SLOT>
R dualmode_engine_t<INCOMING, OUTGOING>::foreach_edges(PULL_SIGNAL&& signal, PULL_SLOT&& slot) {
  if (false == reversed_) {
    return dualmode_detail::__foreach_edges<MSG, R, incoming_graph_t>(
        std::forward<PULL_SIGNAL>(signal), std::forward<PULL_SLOT>(slot), *in_edges_);
  } else {
    return dualmode_detail::__foreach_edges<MSG, R, outgoing_graph_t>(
        std::forward<PULL_SIGNAL>(signal), std::forward<PULL_SLOT>(slot), *out_edges_);
  }
}

template <typename INCOMING, typename OUTGOING>
template <typename MSG, typename R, typename PUSH_SIGNAL, typename PUSH_SLOT>
R dualmode_engine_t<INCOMING, OUTGOING>::foreach_edges(PUSH_SIGNAL&& signal, PUSH_SLOT&& slot,
    dualmode_engine_t<INCOMING, OUTGOING>::v_subset_t& actives) {
  if (false == reversed_) {
    return dualmode_detail::__foreach_edges<MSG, R, outgoing_graph_t>(
        std::forward<PUSH_SIGNAL>(signal), std::forward<PUSH_SLOT>(slot), *out_edges_, actives);
  } else {
    return dualmode_detail::__foreach_edges<MSG, R, incoming_graph_t>(
        std::forward<PUSH_SIGNAL>(signal), std::forward<PUSH_SLOT>(slot), *in_edges_, actives);
  }
}

// ******************************************************************************* //

}  // namespace plato

#endif

