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

#ifndef __PLATO_ALGO_INFOMAP_HPP__
#define __PLATO_ALGO_INFOMAP_HPP__

#include <cstdint>
#include <cstdlib>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/util/spinlock.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"

#include <parallel/algorithm>

namespace plato { namespace algo {

struct infomap_opts_t {
  std::string input_ = "";
  std::string output_ = "";
  int alpha_ = -1;
  bool is_directed_ = false;
  bool need_encode_ = false;
  int pagerank_iter_ = 50;
  double pagerank_threshold_ = 0.0001; //the calculation will be considered complete if the sum of
                                      //the difference of the 'rank' value between iterations
                                      // changes less than threshold.
  double teleport_prob_ = 0.15; //directed graph only
  int inner_iter_ = 3;
  int outer_iter_ = 2;
  int comm_info_num_ = 100; // output some top comm size info
};

using infomap_graph_t = bcsr_t<double, plato::sequence_balanced_by_source_t>;

class infomap_epoch_t {
public:

  explicit infomap_epoch_t(std::shared_ptr<infomap_graph_t> graph, 
      const graph_info_t& graph_info, 
      const infomap_opts_t& opts = infomap_opts_t());

  ~infomap_epoch_t();

  void compute(std::vector<double>& node_flow, std::vector<double>& 
      enter_flow, std::vector<double>& exit_flow);

  std::shared_ptr<infomap_graph_t> graph() { return graph_; };
  std::vector<vid_t>& labels() { return labels_; };

private: 
  inline double plogp(double p) {
    return p > 0.0 ? p * std::log(p) * M_LOG2E : 0.0;
  }

  void calc_code_length();
private:
  std::shared_ptr<infomap_graph_t> graph_;
  std::shared_ptr<infomap_graph_t> rev_graph_;
  graph_info_t graph_info_;
  infomap_opts_t opts_;
  std::vector<vid_t> labels_;
  double tot_enter_flow_;
};

infomap_epoch_t::infomap_epoch_t(std::shared_ptr<infomap_graph_t> graph,
    const graph_info_t& graph_info, const infomap_opts_t& opts):
    graph_(graph), rev_graph_(nullptr), graph_info_(graph_info), 
    opts_(opts) {

  auto& cluster_info = cluster_info_t::get_instance();
  labels_.resize(graph_info_.vertices_);
  #pragma omp parallel for num_threads(cluster_info.threads_)
  for (vid_t i = 0; i < graph_info.vertices_; ++i) {
    labels_[i] = i;
  }

  //build reverse graph
  rev_graph_.reset(new infomap_graph_t(graph_->partitioner()));
  rev_graph_->template load_from_graph<infomap_graph_t>(
      graph_info_, *graph_, false);
}

infomap_epoch_t::~infomap_epoch_t() {

}

void infomap_epoch_t::compute(std::vector<double>& node_flow, 
    std::vector<double>& enter_flow, std::vector<double>& exit_flow) {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::stop_watch_t watch;
  
  tot_enter_flow_ = 0;

  vid_t v_start = graph_->partitioner()->offset_[cluster_info.partition_id_];
  vid_t v_end = graph_->partitioner()->offset_[cluster_info.partition_id_ + 1];
  #pragma omp parallel for num_threads(cluster_info.threads_)
  for (vid_t i = v_start; i < v_end; ++i) {
    auto neighbours = graph_->neighbours(i);
    auto in_neighbours = rev_graph_->neighbours(i);
    if (neighbours.begin_ == neighbours.end_ && 
        in_neighbours.begin_ == in_neighbours.end_) {
      continue;
    }
    write_add(&tot_enter_flow_, enter_flow[i]);
  }

  MPI_Allreduce(MPI_IN_PLACE, &tot_enter_flow_, 1, MPI_DOUBLE, 
      MPI_SUM, MPI_COMM_WORLD);  

  LOG(INFO) << "tot_enter_flow: " << tot_enter_flow_;

  std::vector<spinlock_noaligned_t> locks(HUGESIZE); 

  bitmap_t<> active_all(graph_info_.max_v_i_ + 1);
  active_all.fill();
  auto active_view_all = plato::create_active_v_view(
      graph_->partitioner()->self_v_view(), active_all);
  struct delta_flow_t {
    double delta_exit_;
    double delta_enter_;
  };

  vid_t local_vertex_size = v_end - v_start;
  std::vector<vid_t> fidx(local_vertex_size);

  #pragma omp parallel for num_threads(cluster_info.threads_)
  for (vid_t i = 0; i < local_vertex_size; ++i) {
    fidx[i] = graph_->partitioner()->offset_[cluster_info.partition_id_] + i;
  }

  auto get_delta_code_length = [&](vid_t v_i, vid_t from, vid_t to, 
      double delta_old, double delta_new) {
    double delta_enter = plogp(tot_enter_flow_ + delta_old - delta_new) - 
        plogp(tot_enter_flow_);;

    double delta_log_enter = -plogp(enter_flow[from]) - plogp(enter_flow[to])
        + plogp(enter_flow[from] - enter_flow[v_i] + delta_old)
        + plogp(enter_flow[to] + enter_flow[v_i] - delta_new);

    double delta_log_exit = -plogp(exit_flow[from]) - plogp(exit_flow[to])
        + plogp(exit_flow[from] - exit_flow[v_i] + delta_old)
        + plogp(exit_flow[to] + exit_flow[v_i] - delta_new);

    double delta_log_flow = -plogp(exit_flow[from] + node_flow[from])
        - plogp(exit_flow[to] + node_flow[to])
        + plogp(exit_flow[from] + node_flow[from] - exit_flow[v_i]
        - node_flow[v_i] + delta_old) + plogp(exit_flow[to] + node_flow[to]
        + exit_flow[v_i] + node_flow[v_i] - delta_new);

    return delta_enter - delta_log_enter - delta_log_exit + delta_log_flow;
  };

  struct epoch_msg_type_t {
    vid_t v_i_;
    vid_t from_;
    vid_t to_;
    double flow_;
    double enter_flow_;
    double exit_flow_;
    double delta_old_;
    double delta_new_;
  };

  auto do_change = [&](epoch_msg_type_t& msg) {
    labels_[msg.v_i_] = msg.to_;

    locks[msg.to_ % HUGESIZE].lock();
    node_flow[msg.to_] += msg.flow_;
    enter_flow[msg.to_] += msg.enter_flow_ - msg.delta_new_;
    exit_flow[msg.to_] += msg.exit_flow_ - msg.delta_new_;
    locks[msg.to_ % HUGESIZE].unlock();

    locks[msg.from_ % HUGESIZE].lock();
    node_flow[msg.from_] -= msg.flow_;
    enter_flow[msg.from_] += -msg.enter_flow_ + msg.delta_old_;
    exit_flow[msg.from_] += -msg.exit_flow_ + msg.delta_old_;
    locks[msg.from_ % HUGESIZE].unlock();

    write_add(&tot_enter_flow_, msg.delta_old_ - msg.delta_new_);
  };

  using push_context_t = plato::template mepa_bc_context_t<epoch_msg_type_t>;
  for (int try_time = 0; try_time < opts_.inner_iter_; ++try_time){
    watch.mark("t0");
    __gnu_parallel::random_shuffle(fidx.begin(), fidx.end());
    auto exec_once = [&](std::function<bool(vid_t, vid_t)> condition) {
      plato::broadcast_message<epoch_msg_type_t, vid_t>(      
        active_view_all,
        [&](const push_context_t& context, vid_t v_i) {
          vid_t u = fidx[v_i - v_start];
          auto neighbours = graph_->neighbours(u);
          auto in_neighbours = rev_graph_->neighbours(u);
          if (neighbours.begin_ == neighbours.end_ && 
              in_neighbours.begin_ == in_neighbours.end_) {
            return;
          }
          std::unordered_map<vid_t, delta_flow_t> delta_flow;
          vid_t from = labels_[u];
          //out edge
          for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
            vid_t to = labels_[it->neighbour_];
            if (condition(from, to)) {
              auto p = delta_flow.find(to);
              if (p != delta_flow.end()) {
                p->second.delta_exit_ += it->edata_;
              } else {
                delta_flow.insert(std::make_pair(
                    to, delta_flow_t{it->edata_, 0.0}));
              }
            }
          }

          //in edge
          for (auto it = in_neighbours.begin_; in_neighbours.end_ != it; ++it) {
            vid_t to = labels_[it->neighbour_];
            if (condition(from, to)) {
              auto p = delta_flow.find(to);
              if (p != delta_flow.end()) {
                p->second.delta_enter_ += it->edata_;
              } else {
                delta_flow.insert(std::make_pair(
                    to, delta_flow_t{0.0, it->edata_}));
              }
            }
          }
          
          double best_delta = 0.0;
          vid_t best_mod = (vid_t)-1;
          double delta_old = 0;
          auto p = delta_flow.find(from);
          if (p != delta_flow.end()) {
            delta_old = p->second.delta_exit_ + p->second.delta_enter_;
          }
          double delta_new = 0;
          for (auto e: delta_flow) {
            if (e.first == from) continue;
            double delta = get_delta_code_length(u, from, e.first, 
                delta_old, e.second.delta_exit_ + e.second.delta_enter_);
            if (delta < best_delta) {
              best_mod = e.first;
              best_delta = delta;
              delta_new = e.second.delta_exit_ + e.second.delta_enter_;
            }
          }

          if (best_mod != (vid_t)-1) {
            epoch_msg_type_t msg;
            locks[u % HUGESIZE].lock();
            msg = epoch_msg_type_t{ 
                u, from, best_mod, node_flow[u], enter_flow[u], 
                exit_flow[u], delta_old, delta_new
            };
            locks[u % HUGESIZE].unlock();
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
    exec_once([&](vid_t from, vid_t to){
      if (from <= to) return true;
      return false;
    });
    exec_once([&](vid_t from, vid_t to){
      if (from >= to) return true;
      return false;
    });
    LOG(INFO) << "try_time: "  << try_time << " cost: " << watch.show("t0") / 1000.0;
  }
}

template<typename VID_T = vid_t>
class infomap_t {
public:
  explicit infomap_t(const infomap_opts_t& opts = infomap_opts_t());

  ~infomap_t();

  void compute();

  void output();

  void statistic();

private:

  void calculate_flow();  //load edges and calculate flow

  void init_enter_exit_flow();

  void update_local_label(std::vector<vid_t>& labels);

  std::shared_ptr<infomap_graph_t> rebuild(std::shared_ptr<infomap_graph_t> 
      graph, std::vector<vid_t>& labels);
private:
  infomap_opts_t opts_;
  std::shared_ptr<edge_cache_t<double, vid_t>> cache_;
  graph_info_t graph_info_;
  std::shared_ptr<sequence_balanced_by_source_t> part_bcsr_;
  std::vector<double> node_flow_;
  std::vector<double> exit_flow_;
  std::vector<double> enter_flow_;
  std::vector<vid_t> local_labels_;
  vid_encoder_t<double, VID_T, edge_cache_t> data_encoder_;
};

template<typename VID_T>
infomap_t<VID_T>::infomap_t(const infomap_opts_t& opts) : opts_(opts), 
    cache_(nullptr), graph_info_(opts.is_directed_), part_bcsr_(nullptr) {
}

template<typename VID_T>
infomap_t<VID_T>::~infomap_t() {

}

template<typename VID_T>
void infomap_t<VID_T>::compute() {
  auto& cluster_info = cluster_info_t::get_instance();
  plato::stop_watch_t watch;
  calculate_flow();
  init_enter_exit_flow();

  std::shared_ptr<infomap_graph_t> pgraph(new infomap_graph_t(part_bcsr_));
  graph_info_t graph_info_next(graph_info_);
  graph_info_next.is_directed_ = true;
  pgraph->load_from_cache(graph_info_next, *cache_);
  cache_.reset();

  vid_t local_vertex_size = part_bcsr_->offset_[cluster_info.partition_id_ + 1]
      - part_bcsr_->offset_[cluster_info.partition_id_];
  local_labels_.resize(local_vertex_size);
  #pragma omp parallel for num_threads(cluster_info.threads_)
  for (vid_t i = 0; i < local_vertex_size; ++i) {
    local_labels_[i] = part_bcsr_->offset_[cluster_info.partition_id_] + i;
  }

  infomap_epoch_t* cur = new infomap_epoch_t(pgraph, graph_info_next, opts_);
  for (int epoch = 0; epoch < opts_.outer_iter_; ++epoch) {
    LOG(INFO) << "epoch " << epoch << " begin!";
    watch.mark("t1");
    cur->compute(node_flow_, enter_flow_, exit_flow_);
    update_local_label(cur->labels());
    LOG(INFO) << "epoch " << epoch << " end!, cost: " << watch.show("t1") / 1000.0;
    if (epoch == opts_.outer_iter_ - 1) {
      delete cur;
      break;
    }

    auto graph_next = rebuild(cur->graph(), cur->labels());
    infomap_epoch_t* nxt = new infomap_epoch_t(graph_next, graph_info_next, opts_);
    delete cur;
    cur = nxt; 
  }
}

template<typename VID_T>
void infomap_t<VID_T>::output() {
  auto& cluster_info = cluster_info_t::get_instance();

  fs_mt_omp_output_t fs_out_stream(opts_.output_, (boost::format("%04d") % cluster_info.partition_id_).str(), false);

  vid_t v_start = part_bcsr_->offset_[cluster_info.partition_id_];
  #pragma omp parallel num_threads(cluster_info.threads_)
  {
    int thread_id = omp_get_thread_num();
    int thread_num = omp_get_num_threads();
    auto& fs_output = fs_out_stream.ostream(thread_id);
    for (vid_t i = thread_id; i < (size_t)local_labels_.size(); 
        i += thread_num) {
      vid_t vid = v_start + i;
      vid_t fa = local_labels_[i];
      if (opts_.need_encode_) {
        vid = data_encoder_.decode(vid);
        fa = data_encoder_.decode(fa);
      }
      fs_output << vid << ',' << fa << "\n";
    }
  }

}

template<typename VID_T>
void infomap_t<VID_T>::statistic() {
  //output community info
  auto& cluster_info = cluster_info_t::get_instance();
  std::vector<int> num(graph_info_.vertices_, 0);
  #pragma omp parallel for num_threads(cluster_info.threads_)
  for (vid_t i = 0; i < (vid_t)local_labels_.size(); i++) {
    __sync_fetch_and_add(&num[local_labels_[i]], 1);
  }
  MPI_Allreduce(MPI_IN_PLACE, num.data(), num.size(), MPI_INT, 
      MPI_SUM, MPI_COMM_WORLD);  
  
  int comm_num = 0;
  #pragma omp parallel for num_threads(cluster_info.threads_)
  for (vid_t i = 0; i < graph_info_.vertices_; i++) {
    if (num[i] != 0) {
      __sync_fetch_and_add(&comm_num, 1);
    }
  }

  int idx = 0;
  std::vector<std::pair<int, int> > comm_info(comm_num);
  #pragma omp parallel for num_threads(cluster_info.threads_)
  for (vid_t i = 0; i < graph_info_.vertices_; i++) {
    if (num[i] != 0) {
      int cur = __sync_fetch_and_add(&idx, 1);
      comm_info[cur] = std::make_pair(num[i], i);
    }
  }
  
  __gnu_parallel::sort(comm_info.begin(), comm_info.end());
  int info_num = std::min(opts_.comm_info_num_, comm_num);

  if (cluster_info.partition_id_ == 0) {
    LOG(INFO) << "total comm num: " << comm_num;
    for (int i = 0; i < info_num; ++i) {
      vid_t comm = comm_info[comm_num - 1 - i].second;
      if (opts_.need_encode_) {
        comm = data_encoder_.decode(comm);
      }
      LOG(INFO) << "comm: " << comm <<
          " num:" << comm_info[comm_num - 1 - i].first;
    }
  }
}

template<typename VID_T>
void infomap_t<VID_T>::calculate_flow() { //load cache and calculate flow
  auto& cluster_info = cluster_info_t::get_instance();
  plato::stop_watch_t watch;
  watch.mark("t0");
  decoder_with_default_t<double> decoder((double)1);
  auto encoder_ptr = &data_encoder_;
  if (!opts_.need_encode_) encoder_ptr = nullptr;
  double sum_self_link_weight = 0;
  double sum_link_weight = 0;
  double sum_undir_link_weight = 0;
  auto callback = [&](edge_unit_t<double, vid_t>* input, size_t size) {
    double self_link_weight = 0;
    double link_weight = 0;
    for (size_t i = 0; i < size; ++i) {
      auto& edge = input[i];
      if (edge.src_ == edge.dst_) {
        self_link_weight += edge.edata_;
      } 
      link_weight += edge.edata_;
    }
    write_add(&sum_self_link_weight, self_link_weight);
    write_add(&sum_link_weight, link_weight);
    return true;
  };

  cache_ = load_edges_cache<double, VID_T, edge_cache_t>(&graph_info_,
      opts_.input_, edge_format_t::CSV, decoder, callback, encoder_ptr);

  MPI_Allreduce(MPI_IN_PLACE, &sum_self_link_weight, 1, MPI_DOUBLE, 
      MPI_SUM, MPI_COMM_WORLD);  
  MPI_Allreduce(MPI_IN_PLACE, &sum_link_weight, 1, MPI_DOUBLE, 
      MPI_SUM, MPI_COMM_WORLD);  

  if (cluster_info.partition_id_ == 0) {
    LOG(INFO) << "total vertices: " << graph_info_.vertices_;
    LOG(INFO) << "total edges: " << graph_info_.edges_;
    LOG(INFO) << "is directed: " << graph_info_.is_directed_;
    LOG(INFO) << "total self link weight: " << sum_self_link_weight;
    LOG(INFO) << "total link weight: " << sum_link_weight;
  }

  sum_undir_link_weight = 2 * sum_link_weight - sum_self_link_weight;
  std::vector<vid_t> degrees = generate_dense_out_degrees<vid_t>(
      graph_info_, *cache_);
  
  plato::eid_t __edges = graph_info_.edges_;
  if (false == graph_info_.is_directed_) { __edges = __edges * 2; }

  part_bcsr_.reset(new sequence_balanced_by_source_t(degrees.data(), 
        graph_info_.vertices_, __edges, opts_.alpha_));
  part_bcsr_->check_consistency();
  
  node_flow_.resize(graph_info_.vertices_, 0.0);
  std::vector<double> out_weight(graph_info_.vertices_, 0.0);

  // add node flow
  cache_->reset_traversal();
  #pragma omp parallel num_threads(cluster_info.threads_)
  {
    auto traversal = [&](size_t, edge_unit_t<double>* edge) {
      write_add(&out_weight[edge->src_], edge->edata_);
      write_add(&node_flow_[edge->src_], 
          edge->edata_ / sum_undir_link_weight);
      if (edge->src_ != edge->dst_) {
        if (opts_.is_directed_ == false) {
          write_add(&out_weight[edge->dst_], edge->edata_);
        }
        write_add(&node_flow_[edge->dst_], 
            edge->edata_ / sum_undir_link_weight);
      }
      return true;
    };
    size_t chunk_size = 64;
    while (cache_->next_chunk(traversal, &chunk_size)) { }
  }

  MPI_Allreduce(MPI_IN_PLACE, node_flow_.data(), graph_info_.vertices_, 
      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, out_weight.data(), graph_info_.vertices_, 
      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  if (graph_info_.is_directed_ == false) {
    cache_->reset_traversal();
    #pragma omp parallel num_threads(cluster_info.threads_)
    {
      auto traversal = [&](size_t, edge_unit_t<double>* edge) {
        edge->edata_ = edge->edata_ / (sum_undir_link_weight / 2);
        return true;
      };
      size_t chunk_size = 64;
      while (cache_->next_chunk(traversal, &chunk_size)) { }
    }
  } else { //directed graph need calculate teleport rates 

    LOG(INFO) << "is directed graph, need pagerank";
    std::vector<double> teleport_rate(graph_info_.vertices_, 0.0);
    cache_->reset_traversal();
    #pragma omp parallel num_threads(cluster_info.threads_)
    {
      auto traversal = [&](size_t, edge_unit_t<double>* edge) {
        write_add(&teleport_rate[edge->src_], 
            edge->edata_ / sum_link_weight);
        if (out_weight[edge->src_] > 0) {
          edge->edata_ = edge->edata_ / out_weight[edge->src_];
        }
        return true;
      };
      size_t chunk_size = 64;
      while (cache_->next_chunk(traversal, &chunk_size)) { }
    }
    MPI_Allreduce(MPI_IN_PLACE, teleport_rate.data(), graph_info_.vertices_, 
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    vid_t dangling_num = 0;
    #pragma omp parallel for num_threads(cluster_info.threads_)
    for (vid_t i = 0; i < graph_info_.vertices_; ++i) {
      if (degrees[i] == 0) {
        write_add(&dangling_num, (vid_t)1);
      }
    }

    LOG(INFO) << "dangling num: " << dangling_num;

    vid_t cur_idx = 0;
    std::vector<vid_t> danglings(dangling_num + 1);
    #pragma omp parallel for num_threads(cluster_info.threads_)
    for (vid_t i = 0; i < graph_info_.vertices_; ++i) {
      if (degrees[i] == 0) {
        vid_t idx = __sync_fetch_and_add(&cur_idx, (vid_t)1);
        danglings[idx] = i;
      }
    }

    std::vector<double> node_flow_tmp;
    node_flow_tmp.swap(out_weight);

    int iter = 0;
    double alpha = opts_.teleport_prob_;
    double beta = 1.0 - alpha;
    double sqdiff = 1.0;
    double dangling_rank = 0;
    watch.mark("t2");
    do {

      LOG(INFO) << "pagerank-iter: " << iter; 
      dangling_rank = 0;
      #pragma omp parallel for num_threads(cluster_info.threads_)
      for (vid_t i = 0; i < dangling_num; ++i) {
        write_add(&dangling_rank, node_flow_[danglings[i]]);
      }
      
      #pragma omp parallel for num_threads(cluster_info.threads_)
      for (vid_t i = 0; i < graph_info_.vertices_; ++i) {
        node_flow_tmp[i] = 0.0;
      }

      cache_->reset_traversal();
      #pragma omp parallel num_threads(cluster_info.threads_)
      {
        auto traversal = [&](size_t, edge_unit_t<double>* edge) {
          write_add(&node_flow_tmp[edge->dst_],
              beta * edge->edata_ * node_flow_[edge->src_]);
          return true;
        };
        size_t chunk_size = 64;
        while (cache_->next_chunk(traversal, &chunk_size)) { }
      }

      MPI_Allreduce(MPI_IN_PLACE, node_flow_tmp.data(), graph_info_.vertices_, 
          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      double sum = 0;
      double sqdiff_old = sqdiff;
      sqdiff = 0;
      #pragma omp parallel for num_threads(cluster_info.threads_)
      for (vid_t i = 0; i < graph_info_.vertices_; ++i) {
        node_flow_tmp[i] += (alpha + beta * dangling_rank) * teleport_rate[i];
        write_add(&sum, node_flow_tmp[i]);
        write_add(&sqdiff, std::fabs(node_flow_[i] - node_flow_tmp[i]));
        node_flow_[i] = node_flow_tmp[i];
      }

      if (std::fabs(sum - 1.0) > 1e-10) { //normalize
        #pragma omp parallel for num_threads(cluster_info.threads_)
        for (vid_t i = 0; i < graph_info_.vertices_; ++i) {
          node_flow_[i] /= sum;
        }
      }

      if (sqdiff == sqdiff_old) {
        alpha += 1e-10;
        beta = 1.0 - alpha;
      }

      iter++;

    } while(iter < opts_.pagerank_iter_ && sqdiff > opts_.pagerank_threshold_);

    LOG(INFO) << "pagerank cost: " << watch.show("t2") / 1000.0 << 
      " actual iteration: " << iter;

    #pragma omp parallel for num_threads(cluster_info.threads_)
    for (vid_t i = 0; i < graph_info_.vertices_; ++i) {
      node_flow_[i] = 0.0;
    }

    cache_->reset_traversal();
    #pragma omp parallel num_threads(cluster_info.threads_)
    {
      auto traversal = [&](size_t, edge_unit_t<double>* edge) {
        write_add(&node_flow_[edge->dst_], edge->edata_ * 
            node_flow_tmp[edge->src_] / (1.0 - dangling_rank));
        return true;
      };
      size_t chunk_size = 64;
      while (cache_->next_chunk(traversal, &chunk_size)) { }
    }
    MPI_Allreduce(MPI_IN_PLACE, node_flow_.data(), graph_info_.vertices_, 
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    cache_->reset_traversal();
    #pragma omp parallel num_threads(cluster_info.threads_)
    {
      auto traversal = [&](size_t, edge_unit_t<double>* edge) {
        edge->edata_ *= node_flow_tmp[edge->src_] / (1.0 - dangling_rank);
        return true;
      };
      size_t chunk_size = 64;
      while (cache_->next_chunk(traversal, &chunk_size)) { }
    }
  }

  LOG(INFO) << "calculate flow cost: " << watch.show("t0") / 1000.0;

}

template<typename VID_T>
void infomap_t<VID_T>::init_enter_exit_flow() {
  auto& cluster_info = cluster_info_t::get_instance();
  enter_flow_.resize(graph_info_.vertices_, 0.0);
  exit_flow_.resize(graph_info_.vertices_, 0.0);

  cache_->reset_traversal();
  #pragma omp parallel num_threads(cluster_info.threads_)
  {
    auto traversal = [&](size_t, edge_unit_t<double>* edge) {
      if (edge->dst_ != edge->src_) {
        if (opts_.is_directed_) {
          write_add(&enter_flow_[edge->dst_], edge->edata_);
          write_add(&exit_flow_[edge->src_], edge->edata_);
        }
        else {
          double half_flow = 0.5 * edge->edata_;
          write_add(&enter_flow_[edge->src_], half_flow);
          write_add(&exit_flow_[edge->src_], half_flow);
          write_add(&enter_flow_[edge->dst_], half_flow);
          write_add(&exit_flow_[edge->dst_], half_flow);
        }
      }
      return true;
    };
    size_t chunk_size = 64;
    while (cache_->next_chunk(traversal, &chunk_size)) { }
  }
  
  MPI_Allreduce(MPI_IN_PLACE, enter_flow_.data(), graph_info_.vertices_, 
      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, exit_flow_.data(), graph_info_.vertices_, 
      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

template<typename VID_T>
void infomap_t<VID_T>::update_local_label(std::vector<vid_t>& labels) {
  auto& cluster_info = cluster_info_t::get_instance();
  #pragma omp parallel for num_threads(cluster_info.threads_)
  for (vid_t i = 0; i < (vid_t)local_labels_.size(); ++i) {
    vid_t val = local_labels_[i];
    if (labels[val] != val) {
      local_labels_[i] = labels[val];
    }
  }
}

template<typename VID_T>
std::shared_ptr<infomap_graph_t> infomap_t<VID_T>::rebuild(
    std::shared_ptr<infomap_graph_t> graph, std::vector<vid_t>& labels) {
  auto& cluster_info = cluster_info_t::get_instance();
  plato::stop_watch_t watch;
  watch.mark("t0");
  watch.mark("t1");
  eid_t total_edge = 0;
  vid_t v_start = graph->partitioner()->offset_[cluster_info.partition_id_];
  vid_t v_end = graph->partitioner()->offset_[cluster_info.partition_id_ + 1];
  #pragma omp parallel for num_threads(cluster_info.threads_)
  for (vid_t i = 0; i < (vid_t)local_labels_.size(); ++i) {
    auto neighbours = graph->neighbours(i + v_start);
    write_add(&total_edge, (eid_t)(neighbours.end_ - neighbours.begin_));
  }

  MPI_Allreduce(MPI_IN_PLACE, &total_edge, 1, get_mpi_data_type<eid_t>(), 
      MPI_SUM, MPI_COMM_WORLD);
  LOG(INFO) << "rebuild before total edge: " << total_edge << " cost: " 
      << watch.show("t1") / 1000.0;

  watch.mark("t1");
  std::vector<eid_t> edge_idx(v_end - v_start + 1, 0);
  std::vector<eid_t> tmp_idx(v_end - v_start + 1, 0);

  bitmap_t<> active_all(graph_info_.max_v_i_ + 1);
  active_all.fill();
  auto active_view_all = plato::create_active_v_view(graph->partitioner()->self_v_view(), active_all);

  {
    //first, calc degree
    struct degree_sync_msg_type_t {
      vid_t src;
      vid_t degree;
    };
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
        vid_t pos = msg.src - v_start + 1;
        plato::write_add(&edge_idx[pos], (eid_t)msg.degree);
        return 0;
      }
    );

    for (int i = 1; i < (int)edge_idx.size(); ++i) {
      edge_idx[i] = edge_idx[i - 1] + edge_idx[i];
      tmp_idx[i] = edge_idx[i];
    }

    LOG(INFO) << "rebuild calc degree cost: " << watch.show("t1") / 1000.0;
  }
  eid_t total_local_edge = edge_idx[edge_idx.size() - 1];
  std::vector<std::pair<vid_t, double> > edges(total_local_edge);
  LOG(INFO) << "rebuild local edge: " << edges.size();
  watch.mark("t1");
  {
    //second, transfer edge
    using push_context_t = plato::template mepa_sd_context_t<edge_unit_t<double>>;
    plato::spread_message<edge_unit_t<double>, vid_t>(
      active_view_all,
      [&](const push_context_t& context, vid_t v_i) {
        auto neighbours = graph->neighbours(v_i);
        if (neighbours.begin_ == neighbours.end_) return;
        vid_t src = labels[v_i];
        auto send_to = graph->partitioner()->get_partition_id(src);
        for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
          vid_t dst = labels[it->neighbour_];
          context.send(send_to, edge_unit_t<double>{ src, dst, it->edata_ });
        }
      },
      [&](edge_unit_t<double>& msg) {
        vid_t pos = msg.src_ - v_start;
        vid_t idx = __sync_fetch_and_add(&tmp_idx[pos], (eid_t)1);
        edges[idx].first = msg.dst_;
        edges[idx].second = msg.edata_;
        return 0;
      }
    );
    LOG(INFO) << "rebuild transfer edge cost: " << watch.show("t1") / 1000.0;
  }

  watch.mark("t1");
  edge_cache_t<double> edge_cache;
  {
    //third, sort and aggregate
    #pragma omp parallel for num_threads(cluster_info.threads_)
    for (vid_t v_i = v_start ; v_i < v_end; ++v_i) {
      vid_t pos = v_i - v_start;
      eid_t e_start = edge_idx[pos];
      eid_t e_end = edge_idx[pos + 1];
      if (e_start == e_end) continue;
      std::sort(edges.begin() + e_start, edges.begin() + e_end);
      vid_t pre = (vid_t)-1;
      double local_sum = 0;
      for (eid_t e = e_start; e < e_end; ++e) {
        if (pre != edges[e].first) {
          if (pre != (vid_t)-1 && v_i != pre) {
            edge_cache.push_back(edge_unit_t<double> { v_i, pre, local_sum });
          }
          pre = edges[e].first;
          local_sum = 0;
        }
        local_sum += edges[e].second;
      }
      if (pre != (vid_t)-1 && v_i != pre) {
        edge_cache.push_back(edge_unit_t<double> { v_i, pre, local_sum });
      }
    }

    eid_t edge_num_new = edge_cache.size();
    MPI_Allreduce(MPI_IN_PLACE, &edge_num_new, 1, get_mpi_data_type<eid_t>(), 
        MPI_SUM, MPI_COMM_WORLD);
    LOG(INFO) << "rebuild new edge num: " << edge_num_new;
    LOG(INFO) << "rebuild aggregate edge cost: " << watch.show("t1") / 1000.0;
  }

  watch.mark("t1");
  graph_info_t graph_info_next(graph_info_);
  graph_info_next.is_directed_ = true;
  std::shared_ptr<infomap_graph_t> pgraph(new infomap_graph_t(graph->partitioner()));
  pgraph->load_from_cache(graph_info_next, edge_cache);
  LOG(INFO) << "rebuild load cache cost: " << watch.show("t1") / 1000.0;

  LOG(INFO) << "rebuild total cost: " << watch.show("t0") / 1000.0;

  return pgraph;
}

}}  // namespace algo, namespace plato

#endif

