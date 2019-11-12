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

#ifndef __PLATO_ALGO_FAST_UNFOLDING_LOUVAIN_DENSITY_HPP__
#define __PLATO_ALGO_FAST_UNFOLDING_LOUVAIN_DENSITY_HPP__

#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <boost/format.hpp>
#include <boost/lockfree/queue.hpp>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"
#include "plato/algo/fast_unfolding/louvain.hpp"

namespace plato { namespace algo {

template<typename GRAPH>
class louvain_density_epoch_t {
public:
  using partition_t = typename GRAPH::partition_t;
  using edge_value_t = typename GRAPH::edata_t;
  using louvain_value_t = plato::dense_state_t<edge_value_t, partition_t>;
  using adj_unit_list_spec_t = typename GRAPH::adj_unit_list_spec_t;
  struct epoch_msg_type_t {
    vid_t v_i;
    vid_t from;
    vid_t to;
    edge_value_t ki;
    edge_value_t ki_in_from;
    edge_value_t ki_in_to;
    uint32_t in_from_edges;
    uint32_t in_to_edges;
  };

  struct sync_val_msg_type_t{
    vid_t v_i;
    edge_value_t val;
    uint32_t edges;
    edge_value_t in_sum;
  };


public:
  /**
   * @brief
   * @param graph
   * @param graph_info
   * @param m
   * @param opts
   */
  explicit louvain_density_epoch_t(
    std::shared_ptr<GRAPH> graph, const graph_info_t& graph_info, double m,
    const louvain_opts_t& opts = louvain_opts_t());

  /**
   * @brief
   */
  ~louvain_density_epoch_t();

  /**
   * @brief
   */
  void compute();

  /**
   * @brief getter
   * @return
   */
  std::shared_ptr<GRAPH> graph() { return graph_; };

  /**
   * @brief getter
   * @return
   */
  std::vector<vid_t>& labels() { return labels_; };

private:
  std::shared_ptr<GRAPH> graph_;
  graph_info_t graph_info_;
  louvain_value_t ki_;
  std::vector<vid_t> labels_;
  std::vector<edge_value_t> sigma_tot_;
  bitmap_t<> local_bit_;
  double m_;
  std::vector<edge_value_t> sigma_in_;
  std::vector<uint32_t>  c_vertices_;
  std::vector<uint32_t>  c_edges_;
  louvain_opts_t opts_;
};

template <typename GRAPH>
louvain_density_epoch_t<GRAPH>::louvain_density_epoch_t(
  std::shared_ptr<GRAPH> graph,
  const graph_info_t& graph_info, double m, const louvain_opts_t& opts) :
  graph_(graph), graph_info_(graph_info), ki_(graph_info.max_v_i_, graph->partitioner()),
  labels_(graph_info.max_v_i_ + 1), sigma_tot_(graph_info.max_v_i_ + 1, 0),
  local_bit_(graph_info.max_v_i_ + 1), m_(2 * m), sigma_in_(graph_info.max_v_i_ + 1, 0),
  c_vertices_(graph_info.max_v_i_ + 1, 1),c_edges_(graph_info.max_v_i_ + 1, 0),opts_(opts) {

  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::stop_watch_t watch;

  {// init labels
    watch.mark("t0");
#pragma omp parallel for num_threads(cluster_info.threads_)
    for (vid_t v_i = 0; v_i <= graph_info.max_v_i_; ++v_i) {
      labels_[v_i] = v_i;
    }
    LOG(INFO) << "epoch init labels: " << watch.show("t0") / 1000.0;
  }

  {//calc ki_ and sync sigma_tot_
    int cnt =0;
    bitmap_t<> active_all(graph_info.max_v_i_ + 1);
    active_all.fill();
    auto active_view_all = plato::create_active_v_view(graph->partitioner()->self_v_view(), active_all);
    watch.mark("t0");
    edge_value_t max_ki = 0;
    using push_context_t = plato::template mepa_bc_context_t<sync_val_msg_type_t>;
    plato::broadcast_message<sync_val_msg_type_t, vid_t>(
      active_view_all,
      [&](const push_context_t& context, vid_t v_i) {
        edge_value_t local_sum = 0;
        edge_value_t self_sum =0;
        uint32_t self_num =0;
        auto neighbours = graph->neighbours(v_i);
        for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
          local_sum += it->edata_;
          if(it->neighbour_ == v_i) {
            self_num++;
            self_sum += it->edata_;
          }
        }
        if (neighbours.begin_ != neighbours.end_) {
          ki_[v_i] = local_sum;
          local_bit_.set_bit(v_i);
          plato::write_max(&max_ki, local_sum);
          context.send(sync_val_msg_type_t{ v_i, local_sum, self_num, self_sum});
        }
      },
      [&](int p_i, sync_val_msg_type_t& msg) {
        sigma_tot_[msg.v_i] = msg.val;
        sigma_in_[msg.v_i]  = msg.in_sum;
        c_edges_[msg.v_i]   = msg.edges;
        return 0;
      }
    );
    LOG(INFO) << "epoch sync sigma_tot" << watch.show("t0") / 1000.0;
    MPI_Allreduce(MPI_IN_PLACE, &max_ki, 1, get_mpi_data_type<edge_value_t>(), MPI_MAX, MPI_COMM_WORLD);
    LOG(INFO) << "max-ki is: " << max_ki;
  }
}

template <typename GRAPH>
louvain_density_epoch_t<GRAPH>::~louvain_density_epoch_t() {
}

template <typename GRAPH>
void louvain_density_epoch_t<GRAPH>::compute() {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  uint32_t graph_vertices= graph_info_.vertices_;
  double p_all = 1.0;
  if(graph_vertices > 1) { p_all = graph_info_.edges_*2.0 / ((graph_vertices - 1) * graph_vertices); }

  auto try_change =
    [&](vid_t v_i, vid_t from, vid_t to, edge_value_t ki_in_from, edge_value_t ki_in_to, int32_t in_from_edges,  int32_t in_to_edges) {
      //p = 2mc / nc(nc -1) - M*2 / N(N - 1)
      //Q = [p*in/2m + p^2 * tot^2 / 4m^2 ]

      double p_from = 1.0 - p_all, p_to = 1.0 - p_all, p_from_change = 1.0 - p_all, p_to_change = 1.0 - p_all;
      if(c_vertices_[from] > 1) {
        p_from = c_edges_[from]*2.0 / (c_vertices_[from] * (c_vertices_[from] - 1)) - p_all;
        if(c_vertices_[from] > 2) { p_from_change= (c_edges_[from] - in_from_edges) * 2.0 / ((c_vertices_[from] - 1) * (c_vertices_[from] - 2)) - p_all;}
      }
      if(c_vertices_[to] > 1) {
        p_to = c_edges_[to] * 2.0 / (c_vertices_[to] * (c_vertices_[to] - 1)) - p_all;
      }
      p_to_change = (c_edges_[to] + in_to_edges) * 2.0 / (c_vertices_[to] * (c_vertices_[to] + 1)) - p_all;

      double x1 = (p_from_change - p_from) * sigma_in_[from] - p_from_change * ki_in_from + (p_to_change - p_to) * sigma_in_[to] + p_to_change * ki_in_to;

      p_from_change *= p_from_change;
      p_from *= p_from;
      p_to_change  *= p_to_change;
      p_to *= p_to;

      double x2 = p_from_change * (sigma_tot_[from] * sigma_tot_[from] - 2 * ki_[v_i] * sigma_tot_[from] + ki_[v_i] * ki_[v_i]) - p_from * sigma_tot_[from] * sigma_tot_[from];
      x2 += p_to_change * (sigma_tot_[to] * sigma_tot_[to] + 2 * ki_[v_i] * sigma_tot_[to] + ki_[v_i] * ki_[v_i]) - p_to * sigma_tot_[to] * sigma_tot_[to];
      return x1 - x2 / m_ ;
    };

  auto do_change =
    [&](epoch_msg_type_t& msg) {
      labels_[msg.v_i] = msg.to;
      plato::write_add(&sigma_tot_[msg.from], -msg.ki);
      plato::write_add(&sigma_tot_[msg.to], msg.ki);

      plato::write_add(&sigma_in_[msg.from], -msg.ki_in_from);
      plato::write_add(&sigma_in_[msg.to], msg.ki_in_to);
      plato::write_add(&c_vertices_[msg.from], (uint32_t)-1);
      plato::write_add(&c_vertices_[msg.to], (uint32_t)1);
      plato::write_add(&c_edges_[msg.from],-msg.in_from_edges);
      plato::write_add(&c_edges_[msg.to],msg.in_to_edges);
    };

  auto active_view = plato::create_active_v_view(graph_->partitioner()->self_v_view(), local_bit_);
  plato::stop_watch_t watch;
  using push_context_t = plato::template mepa_bc_context_t<epoch_msg_type_t>;
  for (int try_time = 0; try_time < opts_.inner_iteration_; ++try_time){
    watch.mark("t0");
    auto exec_once =
      [&](std::function<bool(vid_t, vid_t)> condition) {
        plato::broadcast_message<epoch_msg_type_t, vid_t>(
          active_view,
          [&](const push_context_t& context, plato::vid_t v_i) {
            vid_t target = (vid_t)-1;
            vid_t from = labels_[v_i];
            edge_value_t ki_in_from = 0;
            std::unordered_map<vid_t, edge_value_t> ki_in_map;
            std::unordered_map<vid_t, uint32_t>  ki_in_edges;
            auto neighbours = graph_->neighbours(v_i);
            edge_value_t self_cycle = 0;
            uint32_t self_cycle_edges = 0;
            for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
              vid_t to = labels_[it->neighbour_];
              if (condition(from, to)) {
                ki_in_map[to] += it->edata_;
                ki_in_edges[to]++;
              } else if (to == from) {
                ki_in_from += it->edata_;
                ki_in_edges[from]++;
              }
              if (it->neighbour_ == v_i) {
                self_cycle_edges++;
                self_cycle = it->edata_;
              }
            }
            edge_value_t cur_kto,cur_kfrom;
            double best_delta = 0;
            for (auto& it: ki_in_map) {
              double delta = try_change(v_i, from, it.first, ki_in_from, it.second + self_cycle, ki_in_edges[from], ki_in_edges[it.first] + self_cycle_edges);
              if (delta > best_delta) {
                target = it.first;
                best_delta = delta;
                cur_kfrom = ki_in_from;
                cur_kto   = it.second + self_cycle;
              }
            }

            if (target != (vid_t)-1) {

              epoch_msg_type_t msg = epoch_msg_type_t{ v_i, from, target, ki_[v_i],cur_kfrom,cur_kto,ki_in_edges[from],ki_in_edges[target] + self_cycle_edges};
              context.send(msg);
              do_change(msg);
            }
          },
          [&](int p_i, epoch_msg_type_t& msg) {
            if (p_i != cluster_info.partition_id_) {
              do_change(msg);
            }
            return 0;
          }
        );
      };

    exec_once(
      [&](vid_t from, vid_t to){
      if (from < to) return true;
      return false;
    });
    exec_once(
      [&](vid_t from, vid_t to){
      if (from > to) return true;
      return false;
    });
    LOG(INFO) << "try_time: "  << try_time << " cost: " << watch.show("t0") / 1000.0;
  }
}

template<typename GRAPH>
class louvain_density_fast_unfolding_t {
public:
  using partition_t = typename GRAPH::partition_t;
  using edge_value_t = typename GRAPH::edata_t;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<edge_value_t>;
  using louvain_value_t = plato::dense_state_t<vid_t, partition_t>;

  struct edge_sync_msg_type_t {
    vid_t src;
    vid_t dst;
    edge_value_t data;
  };

  struct degree_sync_msg_type_t {
    vid_t src;
    vid_t degree;
  };

public:
  /**
   * @brief
   * @param graph
   * @param graph_info
   * @param opts
   */
  explicit louvain_density_fast_unfolding_t(
    std::shared_ptr<GRAPH> graph, const graph_info_t& graph_info,
    const louvain_opts_t& opts = louvain_opts_t());

  ~louvain_density_fast_unfolding_t();

  /**
   * @brief
   */
  void compute();

  /**
   * @brief
   * @tparam STREAM
   * @param streams
   */
  template <typename STREAM>
  void save(std::vector<STREAM*>& streams);

private:
  /**
   * @brief
   * @param graph
   * @param labels
   * @return
   */
  std::shared_ptr<GRAPH> rebuild(std::shared_ptr<GRAPH> graph, std::vector<vid_t>& labels);

  /**
   * @brief
   * @param labels
   */
  void update_local_label(std::vector<vid_t>& labels);
private:
  std::shared_ptr<GRAPH> graph_;
  graph_info_t graph_info_;
  graph_info_t cur_graph_info_;
  louvain_value_t local_label_;
  louvain_value_t local_comm_size_;
  louvain_opts_t opts_;
};

template<typename GRAPH>
louvain_density_fast_unfolding_t<GRAPH>::louvain_density_fast_unfolding_t(
  std::shared_ptr<GRAPH> graph, const graph_info_t& graph_info,
  const louvain_opts_t& opts)
  : graph_(graph), graph_info_(graph_info), cur_graph_info_(graph_info),
    local_label_(graph_info.max_v_i_, graph->partitioner()),
    local_comm_size_(graph_info.max_v_i_, graph->partitioner()), opts_(opts) {
}

template<typename GRAPH>
louvain_density_fast_unfolding_t<GRAPH>::~louvain_density_fast_unfolding_t() {
}

template<typename GRAPH>
void louvain_density_fast_unfolding_t<GRAPH>::compute() {
  //first calc m and init label
  plato::stop_watch_t watch;
  watch.mark("t0");
  double m = local_label_.template foreach<double> (
    [&](vid_t v_i, vid_t* pval) {
      *pval = v_i;
      double local_sum = 0;
      auto neighbours = graph_->neighbours(v_i);
      for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
        local_sum += it->edata_;
      }
      return local_sum;
    }
  );
  m /= 2; //undirected
  LOG(INFO) << "calc m: " << m << " cost: " << watch.show("t0") / 1000.0;

  //epoch
  louvain_density_epoch_t<GRAPH>* cur = new louvain_density_epoch_t<GRAPH>(graph_, cur_graph_info_, m, opts_);
  for (int epoch = 0; epoch < opts_.outer_iteration_; epoch++) {
    LOG(INFO) << "epoch " << epoch << " begin!";
    watch.mark("compute");
    cur->compute();
    LOG(INFO) << "compute cost: " << watch.show("compute") / 1000.0;
    update_local_label(cur->labels());
    if (epoch == opts_.outer_iteration_ - 1) {
      delete cur;
      break;
    }
    //rebuild from current graph
    watch.mark("rebuild");
    graph_info_t graph_info_next(cur_graph_info_);
    graph_info_next.is_directed_ = true;
    auto graph_next = rebuild(cur->graph(), cur->labels());
    louvain_density_epoch_t<GRAPH>* nxt = new louvain_density_epoch_t<GRAPH>(graph_next, graph_info_next, m, opts_);
    LOG(INFO) << "rebuild cost: " << watch.show("rebuild") / 1000.0;
    delete cur;
    cur = nxt;
  }
}

template<typename GRAPH>
std::shared_ptr<GRAPH> louvain_density_fast_unfolding_t<GRAPH>::rebuild(std::shared_ptr<GRAPH> graph, std::vector<vid_t>& labels) {
  plato::stop_watch_t watch;
  watch.mark("t0");
  eid_t total_edge = local_label_.template foreach<eid_t> (
    [&](vid_t v_i, vid_t* pval) {
      auto neighbours = graph->neighbours(v_i);
      return (eid_t)(neighbours.end_ - neighbours.begin_);
    }
  );
  LOG(INFO) << "rebuild before total edge: " << total_edge << " cost: " << watch.show("t0") / 1000.0;

  watch.mark("t0");
  auto& cluster_info = plato::cluster_info_t::get_instance();
  vid_t v_begin = graph->partitioner()->offset_[cluster_info.partition_id_];
  vid_t v_end = graph->partitioner()->offset_[cluster_info.partition_id_ + 1];
  std::vector<eid_t> edge_idx(v_end - v_begin + 1, 0);
  std::vector<eid_t> tmp_idx(v_end - v_begin + 1, 0);

  bitmap_t<> active_all(graph_info_.max_v_i_ + 1);
  active_all.fill();
  auto active_view_all = plato::create_active_v_view(graph->partitioner()->self_v_view(), active_all);

  {
    //first, calc degree
    using push_context_t = plato::template mepa_sd_context_t<degree_sync_msg_type_t>;
    plato::spread_message<degree_sync_msg_type_t, vid_t>(
      active_view_all,
      [&](const push_context_t& context, vid_t v_i) {
        auto neighbours = graph->neighbours(v_i);
        if (neighbours.begin_ == neighbours.end_) return;
        vid_t src = labels[v_i];
        auto send_to = graph->partitioner()->get_partition_id(src);
        context.send(send_to, degree_sync_msg_type_t { src, (vid_t)(neighbours.end_ - neighbours.begin_) } );
      },
      [&](degree_sync_msg_type_t& msg) {
        vid_t pos = msg.src - v_begin + 1;
        plato::write_add(&edge_idx[pos], (eid_t)msg.degree);
        return 0;
      }
    );

    for (int i = 1; i < (int)edge_idx.size(); ++i) {
      edge_idx[i] = edge_idx[i - 1] + edge_idx[i];
      tmp_idx[i] = edge_idx[i];
    }

    LOG(INFO) << "rebuild calc degree cost: " << watch.show("t0") / 1000.0;
  }
  eid_t total_local_edge = edge_idx[edge_idx.size() - 1];
  std::vector<std::pair<vid_t, edge_value_t> > edges(total_local_edge);
  LOG(INFO) << "rebuild local edge: " << edges.size();
  watch.mark("t0");
  {
    //second, transfer edge
    using push_context_t = plato::template mepa_sd_context_t<edge_sync_msg_type_t>;
    plato::spread_message<edge_sync_msg_type_t, vid_t>(
      active_view_all,
      [&](const push_context_t& context, vid_t v_i) {
        auto neighbours = graph->neighbours(v_i);
        if (neighbours.begin_ == neighbours.end_) return;
        vid_t src = labels[v_i];
        auto send_to = graph->partitioner()->get_partition_id(src);
        for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
          vid_t dst = labels[it->neighbour_];
          context.send(send_to, edge_sync_msg_type_t{ src, dst, it->edata_ });
        }
      },
      [&](edge_sync_msg_type_t& msg) {
        vid_t pos = msg.src - v_begin;
        vid_t idx = __sync_fetch_and_add(&tmp_idx[pos], (eid_t)1);
        edges[idx].first = msg.dst;
        edges[idx].second = msg.data;
        return 0;
      }
    );
    LOG(INFO) << "rebuild transfer edge cost: " << watch.show("t0") / 1000.0;
  }

  watch.mark("t0");
  bitmap_t<> v_bitmap(std::numeric_limits<vid_t>::max());
  edge_cache_t<edge_value_t> edge_cache;
  {
    //third, sort and aggregate
#pragma omp parallel for num_threads(cluster_info.threads_)
    for (vid_t v_i = v_begin; v_i < v_end; ++v_i) {
      vid_t pos = v_i - v_begin;
      eid_t e_start = edge_idx[pos];
      eid_t e_end = edge_idx[pos + 1];
      if (e_start == e_end) continue;
      std::sort(edges.begin() + e_start, edges.begin() + e_end);
      vid_t pre = (vid_t)-1;
      edge_value_t local_sum = 0;
      for (eid_t e = e_start; e < e_end; ++e) {
        if (pre != edges[e].first) {
          if (pre != (vid_t)-1) {
            v_bitmap.set_bit(v_i);v_bitmap.set_bit(pre);
            edge_cache.push_back(edge_unit_t<edge_value_t> { v_i, pre, local_sum });
          }
          pre = edges[e].first;
          local_sum = 0;
        }
        local_sum += edges[e].second;
      }
      if (pre != (vid_t)-1) {
        v_bitmap.set_bit(v_i);v_bitmap.set_bit(pre);
        edge_cache.push_back(edge_unit_t<edge_value_t> { v_i, pre, local_sum });
      }
    }

    eid_t edge_num_new = edge_cache.size();
    MPI_Allreduce(MPI_IN_PLACE, &edge_num_new, 1, get_mpi_data_type<eid_t>(), MPI_SUM, MPI_COMM_WORLD);
    LOG(INFO) << "rebuild new edge num: " << edge_num_new;
    cur_graph_info_.edges_ = edge_num_new;
    LOG(INFO) << "rebuild aggregate edge cost: " << watch.show("t0") / 1000.0;
  }

  watch.mark("t0");
  graph_info_t graph_info_next(graph_info_);
  graph_info_next.is_directed_ = true;
  std::shared_ptr<GRAPH> pgraph(new GRAPH(graph->partitioner()));
  pgraph->load_from_cache(graph_info_next, edge_cache);
  LOG(INFO) << "rebuild load cache cost is: " << watch.show("t0") / 1000.0;

  cur_graph_info_.vertices_ = v_bitmap.count();
  return pgraph;
}

template<typename GRAPH>
void louvain_density_fast_unfolding_t<GRAPH>::update_local_label(std::vector<vid_t>& labels) {
  local_label_.template foreach<int> (
    [&] (vid_t v_i, vid_t* pval) {
      if (labels[*pval] != *pval) {
        *pval = labels[*pval];
      }
      return 0;
    }
  );
}

template<typename GRAPH>
template<typename STREAM>
void louvain_density_fast_unfolding_t<GRAPH>::save(std::vector<STREAM*>& ss) {
  struct louvain_msg_type_t {
    vid_t src;
    vid_t label;
  };
  boost::lockfree::queue<louvain_msg_type_t> que(1024);
  LOG_IF(FATAL, !que.is_lock_free())
  << "boost::lockfree::queue is not lock free\n";

  // start a thread to pop and edge and write to output
  std::atomic<bool> done(false);
  std::thread pop_write([&done, &ss, &que](void) {
#pragma omp parallel num_threads(ss.size())
    {
      int tid = omp_get_thread_num();
      louvain_msg_type_t vb;
      while (!done) {
        if (que.pop(vb)) {
          *ss[tid] << vb.src << "," << vb.label << "\n";
        }
      }

      while (que.pop(vb)) {
        *ss[tid] << vb.src << "," << vb.label << "\n";
      }
    }
  });

  // traverse
  local_label_.template foreach<int> (
    [&] (vid_t v_i, vid_t* pval) {
      while (!que.push(louvain_msg_type_t {v_i, *pval} )) {
      }
      return 0;
    }
  );

  done = true;
  pop_write.join();
}

}  // namespace plato
}  // namespace algo
#endif
