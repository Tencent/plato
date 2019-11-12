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

#ifndef __PLATO_ALGO_BNC_BADER_HPP__
#define __PLATO_ALGO_BNC_BADER_HPP__

#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <vector>
#include <unordered_set>
#include <boost/format.hpp>
#include <boost/lockfree/queue.hpp>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"

#include "cgm.hpp"

namespace plato { namespace algo {
/**
 * @brief bader option
 */
struct bader_opts_t {
  vid_t max_iteration_ = 0;
  int chosen_ = -1;
  float constant_ = 2;
};

/**
 * @brief
 * @tparam INCOMING
 * @tparam OUTGOING
 * @tparam T
 */
template <typename INCOMING, typename OUTGOING, typename T>
class bader_betweenness_t {
public:
  using partition_t = typename dualmode_detail::partition_traits<INCOMING, OUTGOING>::type;
  using betweenness_state_t  = dense_state_t<T, partition_t>;
  using label_state_t = dense_state_t<vid_t, partition_t>;
  using active_subset_t = bitmap_t<>;
  struct bader_msg_type_t {
    vid_t src_;
    T value_;
  };

  /**
   * @brief
   * @param engine
   * @param graph_info
   * @param opts
   */
  explicit bader_betweenness_t (
    dualmode_engine_t<INCOMING, OUTGOING> * engine,
    const graph_info_t &graph_info, const bader_opts_t& opts = bader_opts_t());
  /// \brief
  ~bader_betweenness_t();

  // return the betweenness centrality of a given vertex
  T get_betweenness_of(vid_t v_i) const;

  // compute the betweenness of all vertices
  void compute();
  /**
   * @brief
   * @tparam STREAM
   * @param streams
   */
  template <typename STREAM>
  void save(std::vector<STREAM*>& streams);

  /**
   * @brief
   * @return
   */
  vid_t get_chosen() const { return chosen_; }

  /**
   * @brief
   * @return
   */
  std::unordered_set<vid_t>* get_samples() { return &samples_; }

  /**
   * @brief
   * @return
   */
  vid_t get_major_componnent_vertices() const { return major_component_vertices_; }

private:
  /**
   * @brief
   * @param root
   * @param post_proc
   */
  void epoch(vid_t root, std::function<void(const betweenness_state_t*, vid_t)> post_proc);

private:
  dualmode_engine_t<INCOMING, OUTGOING> * engine_;
  graph_info_t graph_info_;
  betweenness_state_t betweenness_;
  betweenness_state_t num_paths_;
  betweenness_state_t dependencies_;
  T sum_dependence_;
  T sum_dependence_max_;
  vid_t chosen_;
  float constant_;
  vid_t max_iteration_;
  std::unordered_set<vid_t> samples_;
  connected_component_t<INCOMING, OUTGOING> *cc_;
  label_state_t global_labels_;
  vid_t major_component_label_;
  vid_t major_component_vertices_;
  active_subset_t active_all_;
};

template <typename INCOMING, typename OUTGOING, typename T>
bader_betweenness_t<INCOMING, OUTGOING, T>::bader_betweenness_t(
  dualmode_engine_t<INCOMING, OUTGOING>* engine, const graph_info_t &graph_info,
  const bader_opts_t& opts)
  : engine_(engine),
    graph_info_(graph_info),
    betweenness_(graph_info.max_v_i_, engine->in_edges()->partitioner()),
    num_paths_(graph_info.max_v_i_, engine->in_edges()->partitioner()),
    dependencies_(graph_info.max_v_i_, engine->in_edges()->partitioner()),
    sum_dependence_(0),
    global_labels_(graph_info.max_v_i_, engine->in_edges()->partitioner()),
    active_all_(graph_info.max_v_i_ + 1) {
  cc_ = new connected_component_t<INCOMING, OUTGOING>(engine, graph_info);
  cc_->compute();
  major_component_label_ = cc_->get_major_label();
  major_component_vertices_ = cc_->get_major_vertices();
  constant_ = opts.constant_;
  sum_dependence_max_ = constant_ * major_component_vertices_;
  if (opts.max_iteration_ == 0) {
    max_iteration_ = major_component_vertices_;
  } else {
    max_iteration_ = opts.max_iteration_;
  }

  active_all_.fill();

  //init label
  auto local_label = cc_->get_labels();

  auto active_view_all = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), active_all_);
  active_view_all.template foreach<vid_t>([&](vid_t v_i){
    global_labels_[v_i] = (*local_label)[v_i];
    return 0;
  });

  auto& cluster_info = plato::cluster_info_t::get_instance();
  if (cluster_info.partition_id_ != 0) {
    vid_t v_begin = engine_->out_edges()->partitioner()->offset_[cluster_info.partition_id_];
    vid_t v_end = engine_->out_edges()->partitioner()->offset_[cluster_info.partition_id_+1];
    MPI_Send(&(global_labels_[v_begin]), v_end - v_begin, get_mpi_data_type<vid_t>(), 0, 0, MPI_COMM_WORLD);
  }
  else {
    for (int i = 1; i < cluster_info.partitions_; ++i ) {
      MPI_Status recv_status;
      vid_t v_begin = engine_->out_edges()->partitioner()->offset_[i];
      vid_t v_end = engine_->out_edges()->partitioner()->offset_[i+1];
      MPI_Recv(&(global_labels_[v_begin]), v_end - v_begin, get_mpi_data_type<vid_t>(), i, 0, MPI_COMM_WORLD, &recv_status);
    }
  }

  //get chosen vertex
  if (opts.chosen_ == -1) {
    if (cluster_info.partition_id_ == 0) {
      do {
        chosen_ = rand() % graph_info.vertices_;
      } while (global_labels_[chosen_] != major_component_label_);
    }
    MPI_Bcast(&chosen_, 1, get_mpi_data_type<vid_t>(), 0, MPI_COMM_WORLD);
    LOG(INFO) << "chosen: "  << chosen_ << std::endl;
  }
  else {
    chosen_ = opts.chosen_;
  }

}

template <typename INCOMING, typename OUTGOING, typename T>
bader_betweenness_t<INCOMING, OUTGOING, T>::~bader_betweenness_t() {
  delete cc_;
}

template <typename INCOMING, typename OUTGOING, typename T>
void bader_betweenness_t<INCOMING, OUTGOING, T>::compute() {
  auto continue_iterating = [&](vid_t iter) {
    return iter < max_iteration_ && sum_dependence_ < sum_dependence_max_;
  };

  // pick a new random vertex that is not used before
  // we use a set to record the vertices that are used.
  auto& cluster_info = plato::cluster_info_t::get_instance();
  auto gen_next_vertex = [&]() {
    vid_t nxt;
    if (cluster_info.partition_id_ == 0) {
      size_t curr_size = samples_.size();
      do {
        nxt = std::rand() % graph_info_.vertices_;
        if (global_labels_[nxt] == major_component_label_) {
          samples_.insert(nxt);
        }
      } while (curr_size == samples_.size());
    }
    MPI_Bcast(&nxt, 1, get_mpi_data_type<vid_t>(), 0, MPI_COMM_WORLD);
    if (cluster_info.partition_id_ != 0) samples_.insert(nxt);
    return nxt;
  };
  auto active_view_all = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), active_all_);
  auto accumulate = [&](const betweenness_state_t* dependencies, vid_t root) {
    active_view_all.template foreach<vid_t>([&](vid_t v_i){
      if (v_i != root) {
        T d = (*dependencies)[v_i];
        betweenness_[v_i] += d;
        if (v_i == chosen_) {
          sum_dependence_ += d;
          LOG(INFO) << "chosen add: " << d << std::endl;
        }
      }
      return 1;
    });

    int chosen_partition = engine_->out_edges()->partitioner()->get_partition_id(chosen_);
    MPI_Bcast(&sum_dependence_, 1, get_mpi_data_type<T>(), chosen_partition,MPI_COMM_WORLD);
    LOG(INFO) << "sum_deppendece: " << sum_dependence_ << " sum_dependence_max: " << sum_dependence_max_ << std::endl;
  };

  auto update_betweenness = [&](vid_t sample_size) {
    active_view_all.template foreach<vid_t>([&](vid_t v_i) {
      betweenness_[v_i] = major_component_vertices_ * betweenness_[v_i] / sample_size;
      LOG(INFO) << "vid: " << v_i << " d: " << betweenness_[v_i] << " major: " << major_component_vertices_ << std::endl;
      if (std::isnan(betweenness_[v_i])) {
        betweenness_[v_i] = 0;
      }
      return 1;
    });
  };

  for (vid_t iter = 0; continue_iterating(iter); ++iter) {
    auto root = gen_next_vertex();
    LOG(INFO) << "next root: " << root << std::endl;
    epoch(root, accumulate);
    LOG_IF(INFO, !cluster_info.partition_id_ && (iter + 1) % 10 == 0)
    << boost::format("iter=%u/%u, sum_dependence[%u]=%.1f/%.1f\n") % (iter + 1) % max_iteration_ % chosen_ % sum_dependence_ % sum_dependence_max_;
  }

  vid_t sample_size = samples_.size();
  MPI_Bcast(&sample_size, 1, get_mpi_data_type<vid_t>(), 0, MPI_COMM_WORLD);

  update_betweenness(sample_size);
}

template <typename INCOMING, typename OUTGOING, typename T>
T bader_betweenness_t<INCOMING, OUTGOING, T>::get_betweenness_of(vid_t v_i) const {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  int vtx_partition = engine_->out_edges()->partitioner()->get_partition_id(v_i);
  T ret = cluster_info.partition_id_ == vtx_partition ? betweenness_[v_i] : 0;
  MPI_Bcast(&ret, 1, get_mpi_data_type<T>(), vtx_partition, MPI_COMM_WORLD);
  return ret;
}

template <typename INCOMING, typename OUTGOING, typename T>
void bader_betweenness_t<INCOMING, OUTGOING, T>::epoch(
  vid_t root, std::function<void(const betweenness_state_t*, vid_t)> post_proc) {
  std::vector<active_subset_t*> levels;
  auto active_current = new active_subset_t(graph_info_.max_v_i_ + 1);
  auto visited = engine_->alloc_v_subset();
  vid_t actives = 1;
  visited.clear();
  visited.set_bit(root);
  active_current->clear();
  active_current->set_bit(root);
  levels.push_back(active_current);
  num_paths_.fill((T)0.0);
  num_paths_[root] = 1.0;

  using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;
  using pull_context_t = plato::template mepa_ag_context_t<bader_msg_type_t>;
  using pull_message_t = plato::template mepa_ag_message_t<bader_msg_type_t>;
  using push_context_t = plato::template mepa_bc_context_t<bader_msg_type_t>;
  for (int epoch_i = 0; actives > 0; ++epoch_i) {
    auto active_next = new active_subset_t(graph_info_.max_v_i_ + 1);
    active_next->clear();
    engine_->template foreach_edges<bader_msg_type_t, vid_t> (
      [&](const push_context_t& context, vid_t v_i) {
        context.send(bader_msg_type_t{ v_i, num_paths_[v_i] });
      },
      [&](int /*p_i*/, bader_msg_type_t& msg) {
        auto neighbours = engine_->out_edges()->neighbours(msg.src_);
        for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
          vid_t dst = it->neighbour_;
          if (!visited.get_bit(dst)) {
            if (num_paths_[dst] == 0) {
              active_next->set_bit(dst);
            }
            write_add(&(num_paths_[dst]), msg.value_);
          }
        }
        return 0;
      },
      [&](const pull_context_t& context, vid_t v_i, const adj_unit_list_spec_t& adjs) {
        if (visited.get_bit(v_i)) return;
        T sum = 0;
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          vid_t src = it->neighbour_;
          if (active_current->get_bit(src)) {
            sum += num_paths_[src];
          }
        }
        if (sum > 0) {
          context.send(pull_message_t{ v_i, bader_msg_type_t{ v_i, sum } });
        }
      },
      [&](int, pull_message_t& msg) {
        if (!visited.get_bit(msg.v_i_)) {
          active_next->set_bit(msg.v_i_);
          write_add(&(num_paths_[msg.v_i_]), msg.message_.value_);
        }
        return 0;
      },
      *active_current
    );
    auto active_view = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), *active_next);
    actives = active_view.template foreach<vid_t> ([&](vid_t v_i) {
      visited.set_bit(v_i); return 1;
    });
    if (actives > 0) levels.push_back(active_next);
    active_current = active_next;
  }

  dependencies_.fill((T)0.0);
  visited.clear();
  while (levels.size() > 0) {
    auto active_view = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), *(levels.back()));
    actives = active_view.template foreach<vid_t> ([&](vid_t v_i) {
      visited.set_bit(v_i); return 1;
    });
    engine_->template foreach_edges<bader_msg_type_t, vid_t> (
      [&](const push_context_t& context, vid_t v_i) {
        T value = (dependencies_[v_i] + 1.0) / num_paths_[v_i];
        context.send(bader_msg_type_t{ v_i, value });
      },
      [&](int /*p_i*/, bader_msg_type_t& msg) {
        auto neighbours = engine_->out_edges()->neighbours(msg.src_);
        for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
          vid_t dst = it->neighbour_;
          if (!visited.get_bit(dst)) {
            write_add(&(dependencies_[dst]), msg.value_ * num_paths_[dst]);
          }
        }
        return 0;
      },
      [&](const pull_context_t& context, vid_t v_i, const adj_unit_list_spec_t& adjs) {
        if (visited.get_bit(v_i)) return;
        T sum = 0;
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          vid_t src = it->neighbour_;
          if (levels.back()->get_bit(src)) {
            sum += (dependencies_[src] + 1.0) / num_paths_[src];
          }
        }
        LOG(INFO) << "send sum: " << sum << std::endl;
        if (sum > 0) {
          context.send(pull_message_t{ v_i, bader_msg_type_t{ v_i, sum } });
        }
      },
      [&](int, pull_message_t& msg) {
        if (!visited.get_bit(msg.v_i_)) {
          write_add(&(dependencies_[msg.v_i_]), msg.message_.value_ * num_paths_[msg.v_i_]);
          LOG(INFO) << "vid: " << msg.v_i_ << " recv sum: " << msg.message_.value_ << " num_paths: " << num_paths_[msg.v_i_] << " cur: "
                    << dependencies_[msg.v_i_] << std::endl;
        }
        return 0;
      },
      *(levels.back())
    );
    delete levels.back();
    levels.pop_back();
  }

  post_proc(&dependencies_, root);
}

template <typename INCOMING, typename OUTGOING, typename T>
template <typename STREAM>
void bader_betweenness_t<INCOMING, OUTGOING, T>::save(std::vector<STREAM*>& ss) {
  boost::lockfree::queue<bader_msg_type_t> que(1024);
  LOG_IF(FATAL, !que.is_lock_free()) << "boost::lockfree::queue is not lock free\n";

  // start a thread to pop and edge and write to output
  std::atomic<bool> done(false);
  std::thread pop_write([&done, &ss, &que](void) {
#pragma omp parallel num_threads(ss.size())
    {
      int tid = omp_get_thread_num();
      bader_msg_type_t vb;
      while (!done) {
        if (que.pop(vb)) {
          *ss[tid] << vb.src_ << "," << vb.value_ << "\n";
        }
      }

      while (que.pop(vb)) {
        *ss[tid] << vb.src_ << "," << vb.value_ << "\n";
      }
    }
  });

  // traverse
  auto active_view_all = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), active_all_);
  active_view_all.template foreach<vid_t>([&](vid_t v_i){
    while (!que.push(bader_msg_type_t {v_i, betweenness_[v_i] })) {
    }
    return 1;
  });

  done = true;
  pop_write.join();

}

}
}
#endif
