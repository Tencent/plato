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

#ifndef __PLATO_ALGO_nstepdegrees_HPP__
#define __PLATO_ALGO_nstepdegrees_HPP__

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <float.h>
#include <string>
#include <boost/lockfree/queue.hpp>
#include <unordered_map>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/util/hyperloglog.hpp"
#include "plato/engine/dualmode.hpp"
#include "plato/util/spinlock.hpp"

namespace plato { namespace algo {

struct nstepdegree_opts_t {
  uint32_t step = 20;   // steps
  std::string type = "both";
  bool is_directed = false;
};

enum class count_type_t {
  OUT_DEGREE = 1,
  IN_DEGREE = 2
};

struct degrees_record_t {
  vid_t in_;
  vid_t out_;
};

template<typename INCOMING, typename OUTGOING, uint32_t BitWidth = 12>
class nstepdegrees_t {
public:
  using vid_t = plato::vid_t;
  using hll = plato::hyperloglog_t<BitWidth>;
  using dcsc_spec_t = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;
  using partition_t = dcsc_spec_t::partition_t;
  using graph_info_t = plato::graph_info_t;

  struct nstepdegrees_with_vid_t {
    vid_t v_i;
    vid_t in_;
    vid_t out_;
  }__attribute__((__packed__));

  template <uint32_t MsgBitWidth>
  struct nstepdegrees_msg_type_t {
    vid_t vtx;
    plato::hyperloglog_t<MsgBitWidth> hll_data;
  }__attribute__((__packed__));

  using v_subset_t = bitmap_t<>;
  using hll_state_t = plato::dense_state_t<hll, partition_t>;

public:
  /**
   * @brief
   * @param engine
   * @param graph_info_
   * @param active_v
   * @param opts_
   */
  explicit nstepdegrees_t(dualmode_engine_t<INCOMING, OUTGOING>* engine,  const graph_info_t& graph_info_,  const v_subset_t& active_v, const nstepdegree_opts_t& opts_ = nstepdegree_opts_t());

  ~nstepdegrees_t() {}

  /**
   * @brief
   * @param in_edges
   * @param out_edges
   */
  void mark_target(INCOMING& in_edges, OUTGOING& out_edges);

  /**
   * @brief
   * @param in_edges
   * @param out_edges
   * @param is_reversed
   */
  void spread(INCOMING& in_edges, OUTGOING& out_edges, bool is_reversed);

  /**
   * @brief
   * @param in_edges
   * @param out_edges
   * @param type
   * @param is_reversed
   */
  void propagate(INCOMING& in_edges, OUTGOING& out_edges, count_type_t type, bool is_reversed);

  /**
   * @brief
   * @param in_edges
   * @param out_edges
   */
  void compute(INCOMING& in_edges, OUTGOING& out_edges);

  /**
   * @brief
   * @tparam STREAM
   * @param ss
   */
  template<typename STREAM>
  void save(std::vector<STREAM*>& ss);

  /**
   * @brief
   */
  void view_coverage();

  /**
   * @brief
   */
  void view_degrees();

  /**
   * @brief getter
   * @return
   */
  std::unordered_map<vid_t, degrees_record_t> get_degrees();

private:
  plato::graph_info_t graph_info_;
  dualmode_engine_t<INCOMING, OUTGOING>* engine_;
  v_subset_t covered_;
  v_subset_t active_in_;
  v_subset_t target_;
  nstepdegree_opts_t opts_;
  std::unordered_map<vid_t, degrees_record_t> degrees_;

};

template<typename INCOMING, typename OUTGOING, uint32_t BitWidth>
nstepdegrees_t<INCOMING, OUTGOING, BitWidth>::nstepdegrees_t(
  dualmode_engine_t<INCOMING, OUTGOING>* engine,
  const plato::graph_info_t& graph_info,
  const v_subset_t& active_v,
  const nstepdegree_opts_t& opts) : engine_(engine), graph_info_(graph_info), opts_(opts) {

  active_in_ = engine_->alloc_v_subset();
  target_ = engine_->alloc_v_subset();
  covered_ = engine_->alloc_v_subset();
  active_in_.fill();
  target_.clear();

  auto active_view = plato::create_active_v_view(engine_->in_edges()->partitioner()->self_v_view(), active_in_);
  active_view.template foreach<plato::vid_t>(
    [&](plato::vid_t v_i) {
      if(active_v.get_bit(v_i)) {
        target_.set_bit(v_i);
        return 1;
      }
      return 0;
    });
}

template<typename INCOMING, typename OUTGOING, uint32_t BitWidth>
void nstepdegrees_t<INCOMING, OUTGOING, BitWidth>::mark_target(INCOMING& in_edges, OUTGOING& out_edges) {
  auto active_mark = engine_->alloc_v_subset();
  active_in_.clear();
  covered_.clear();
  active_mark.fill();

  auto active_view = plato::create_active_v_view(out_edges.partitioner()->self_v_view(), active_mark);
  active_view.template foreach<plato::vid_t>(
    [&](plato::vid_t vtx) {
      if(target_.get_bit(vtx)) {
        covered_.set_bit(vtx);
        active_in_.set_bit(vtx);
      }
      return 0;
    });
}

template<typename INCOMING, typename OUTGOING, uint32_t BitWidth>
void nstepdegrees_t<INCOMING, OUTGOING, BitWidth>::spread( INCOMING& in_edges, OUTGOING& out_edges, bool is_reversed) {
  vid_t actives = 1;
  if(is_reversed) {
    engine_->reverse();
  }

  v_subset_t active_out = engine_->alloc_v_subset();
  for(int i = 0; i < opts_.step && actives > 0; ++i) {
    using push_context_t = plato::template mepa_bc_context_t<plato::vid_t>;
    using pull_context_t = plato::template mepa_ag_context_t<plato::vid_t>;
    using pull_message_t = plato::template mepa_ag_message_t<plato::vid_t>;
    using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;

    active_out.clear();
    actives = engine_->template foreach_edges<vid_t, vid_t> (
      [&] (const push_context_t& context, vid_t src) {
        context.send(src);
      },
      [&] (int /*p_i*/, plato::vid_t& msg) {
        vid_t activated = 0;
        auto neighbours = engine_->out_edges()->neighbours(msg);
        for(auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
          vid_t dst = it->neighbour_;
          if(covered_.get_bit(dst)) {
            continue;
          }
          ++activated;
          active_out.set_bit(dst);
        }
        return activated;
      },
      [&] (const pull_context_t& context, vid_t dst, const adj_unit_list_spec_t& adj) {
        if(covered_.get_bit(dst)) {
          return;
        }
        for(auto it = adj.begin_; adj.end_ != it; ++it) {
          vid_t src = it->neighbour_;
          if(active_in_.get_bit(src)) {
            context.send(pull_message_t {dst, src});
            break;
          }
        }
      },
      [&] (int, pull_message_t& msg) {
        active_out.set_bit(msg.v_i_);
        return 1;
      },
      active_in_);

    auto active_view = plato::create_active_v_view(out_edges.partitioner()->self_v_view(), active_out);
    actives = active_view.template foreach<plato::vid_t>(
      [&](plato::vid_t v_i) {
        covered_.set_bit(v_i); return 1;
      });

    std::swap(active_in_, active_out);
  }
  covered_.sync();

  if(is_reversed) {
    engine_->reverse();
  }
}

template <typename INCOMING, typename OUTGOING, uint32_t BitWidth>
void nstepdegrees_t<INCOMING, OUTGOING, BitWidth>::view_coverage() {
  std::vector<vid_t> res;

  auto& cluster_info = plato::cluster_info_t::get_instance();
  //auto in_edge_ = engine->in_edges();
  auto active_view = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), covered_);
  active_view.template foreach<plato::vid_t>(
    [&](plato::vid_t v_i) {
      res.push_back(v_i); return 1;
    });

  spinlock_noaligned_t mylock;
  mylock.lock_ = 0;
  mylock.lock();
  LOG(INFO) << "covered-------------------------->"  << std::endl;
  //LOG(INFO) << "partition_id:" << cluster_info.partition_id_ << ":";
  for(int i = 0; i < res.size(); ++i) {
    LOG(INFO) << res[i] << ",";
  }
  LOG(INFO) << std::endl;
  mylock.unlock();
}

template <typename INCOMING, typename OUTGOING, uint32_t BitWidth>
void nstepdegrees_t<INCOMING, OUTGOING, BitWidth>::view_degrees() {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  auto active_view = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), target_);
  active_view.template foreach<plato::vid_t>(
    [&](plato::vid_t v_i) {
      LOG(INFO) << "vertexId:" << v_i << "---->" << degrees_[v_i].in_ << "," <<degrees_[v_i].out_;
      return 1;
    });
  LOG(INFO) << std::endl;
}

template <typename INCOMING, typename OUTGOING, uint32_t BitWidth>
std::unordered_map<vid_t, degrees_record_t> nstepdegrees_t<INCOMING, OUTGOING, BitWidth>::get_degrees() {
  return degrees_;
}

template<typename INCOMING, typename OUTGOING, uint32_t BitWidth>
void nstepdegrees_t<INCOMING, OUTGOING, BitWidth>::propagate(INCOMING& in_edges, OUTGOING& out_edges, count_type_t type, bool is_reversed) {
  if(is_reversed) {
    engine_->reverse();
  }
  auto books = engine_->template alloc_v_state<hll>();
  auto delta_books = engine_->template alloc_v_state<hll>();
  auto locks = engine_->template alloc_v_state<spinlock_noaligned_t>();
  vid_t actives = 0;
  active_in_.clear();
  books.template foreach<int> (
    [&] (vid_t vtx, hll* hll_data) {
      hll_data->init();
      hll_data->add(&vtx, sizeof(vid_t));
      delta_books[vtx].init();
      new (&locks[vtx]) spinlock_noaligned_t;
      active_in_.set_bit(vtx);
      plato::write_add(&actives, (vid_t)1);
      return 0;
    },
    &covered_);

  auto active_out = engine_->template alloc_v_subset();
  MPI_Allreduce(MPI_IN_PLACE, &actives, 1, plato::get_mpi_data_type<plato::vid_t>(), MPI_SUM,MPI_COMM_WORLD);
  for(int i = 0; i < opts_.step && actives > 0; ++i) {

    active_out.clear();
    using pull_context_t = plato::template mepa_ag_context_t<nstepdegrees_msg_type_t<BitWidth>>;
    using pull_message_t = plato::template mepa_ag_message_t<nstepdegrees_msg_type_t<BitWidth>>;
    using push_context_t = plato::template mepa_bc_context_t<nstepdegrees_msg_type_t<BitWidth>>;
    using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;
    engine_->template foreach_edges<nstepdegrees_msg_type_t<BitWidth>, int> (
      [&] (const push_context_t& context, vid_t src) {
        context.send(nstepdegrees_msg_type_t<BitWidth> {src, books[src]});
      },
      [&] (int /*p_i*/, nstepdegrees_msg_type_t<BitWidth>& msg) {
        auto neighbours = engine_->out_edges()->neighbours(msg.vtx);
        for(auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
          vid_t dst = it->neighbour_;
          locks[dst].lock();
          delta_books[dst].merge(msg.hll_data);
          locks[dst].unlock();
          active_out.set_bit(dst);
        }
        return 0;
      },
      [&] (const pull_context_t& context, vid_t dst, const adj_unit_list_spec_t& adj) {
        hll agg_nei;
        agg_nei.init();
        for(auto it = adj.begin_; adj.end_ != it; ++it) {
          vid_t src = it->neighbour_;
          agg_nei.merge(books[src]);
        }
        context.send(pull_message_t { dst, nstepdegrees_msg_type_t<BitWidth> { dst, agg_nei }});
      },
      [&] (int, pull_message_t& msg) {
        locks[msg.v_i_].lock();
        delta_books[msg.v_i_].merge(msg.message_.hll_data);
        locks[msg.v_i_].unlock();
        active_out.set_bit(msg.v_i_);
        return 0;
      },
      active_in_);

    active_in_.clear();
    actives = books.template foreach<int> (
      [&] (vid_t vtx, hll* hll_data) {
        if(covered_.get_bit(vtx) && hll_data->merge(delta_books[vtx]) > 0) {
          active_in_.set_bit(vtx);
          return 1;
        }
        return 0;
      },
      &active_out);

    delta_books.template foreach<int> (
      [&] (vid_t vtx, hll* hll_data) {
        hll_data->init();
        return 0;
      },
      &covered_);
  }

  books.template foreach<int> (
    [&](vid_t vtx, hll* hll_data) {
      switch(type) {
        case count_type_t::OUT_DEGREE:
          degrees_[vtx].out_ = static_cast<vid_t>(hll_data->estimate()) - 1;
          break;
        case count_type_t::IN_DEGREE:
          degrees_[vtx].in_ = static_cast<vid_t>(hll_data->estimate()) - 1;
          break;
        default:
          CHECK(false) << "unkown count_type_t: " << (int)type;
      }
      return 0;
    },
    &target_);
  if(is_reversed) {
    engine_->reverse();
  }
}

template<typename INCOMING, typename OUTGOING, uint32_t BitWidth>
void nstepdegrees_t<INCOMING, OUTGOING, BitWidth>::compute(INCOMING& in_edges, OUTGOING& out_edges) {
  covered_ = engine_->alloc_v_subset();

  if(opts_.type == "in" || opts_.type == "both") {
    mark_target(in_edges, out_edges);
    spread(in_edges, out_edges, true);
    propagate(in_edges, out_edges, count_type_t::IN_DEGREE, false);
  }

  if(opts_.type == "out" || opts_.type == "both") {
    if(opts_.type == "out" || opts_.is_directed) {
      mark_target(in_edges, out_edges);
      spread(in_edges, out_edges, false);
      propagate(in_edges, out_edges, count_type_t::OUT_DEGREE, true);
    }
    else {
      for(auto kv : degrees_) {
        kv.second.out_ = kv.second.in_;
      }
    }
  }
}

template<typename INCOMING, typename OUTGOING, uint32_t BitWidth>
template<typename STREAM>
void nstepdegrees_t<INCOMING, OUTGOING, BitWidth>::save(std::vector<STREAM*>& ss) {
  if(engine_->is_reversed()) {
    engine_->reverse();
  }
  auto& cluster_info = plato::cluster_info_t::get_instance();
  vid_t v_begin = engine_->out_edges()->partitioner()->offset_[cluster_info.partition_id_];
  vid_t v_end = engine_->out_edges()->partitioner()->offset_[cluster_info.partition_id_+1];
  size_t vtx_count = v_end - v_begin;

  boost::lockfree::queue<nstepdegrees_with_vid_t> que(vtx_count + 1);
  LOG_IF(FATAL, !que.is_lock_free())
  << "boost::lockfree::queue is not lock free\n";
  std::atomic<bool> done(false);
  std::thread pop_write([&done, &ss, &que](void) {
#pragma omp parallel num_threads(ss.size())
    {
      int tid = omp_get_thread_num();
      nstepdegrees_with_vid_t degree;
      while(!done) {
        if(que.pop(degree)) {
          *ss[tid] << degree.v_i << "," << degree.in_ <<","<< degree.out_ << "\n";
        }
      }

      while(que.pop(degree)) {
        *ss[tid] << degree.v_i << "," << degree.in_ <<","<< degree.out_ << "\n";
      }
    }
  });

  auto active_view = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), target_);
  active_view.template foreach<vid_t>([&](vid_t v_i){
    while(!que.push(nstepdegrees_with_vid_t {v_i, degrees_[v_i].in_, degrees_[v_i].out_})) {}
    return 1;
  });

  done = true;
  pop_write.join();
}

}}

#endif
