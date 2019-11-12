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

#ifndef ALGO_CNC_DAVID_HPP_
#define ALGO_CNC_DAVID_HPP_

#include <glog/logging.h>
#include <algorithm>
#include <boost/format.hpp>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "cgm.hpp"
#include "distance.hpp"

namespace plato { namespace algo {
struct david_opts_t {
    vid_t num_samples_ = 10;
};

template <typename INCOMING, typename OUTGOING>
class david_closeness_t {
public:
  using partition_t = typename dualmode_detail::partition_traits<INCOMING, OUTGOING>::type;
  using closeness_state_t  = dense_state_t<double, partition_t>;

  struct david_msg_type_t {
    vid_t src_;
    double value_;
  };

  /**
   * @brief
   * @param engine
   * @param graph_info
   * @param opts
   */
  explicit david_closeness_t(dualmode_engine_t<INCOMING, OUTGOING> * engine, const graph_info_t& graph_info,
      const david_opts_t& opts = david_opts_t());
  /**
   * @brief
   */
  ~david_closeness_t();

  /**
   * @brief
   */
  void compute();

  /**
   * @brief
   * @param v
   * @return
   */
  double get_closeness_of(vid_t v) const;

  /**
   * @brief
   * @tparam STREAM
   * @param ss
   */
  template<typename STREAM>
  void save(std::vector<STREAM*>& ss);

  /**
   * @brief
   * @return
   */
  std::vector<vid_t>* get_samples() {
    return &samples_;
  }

  /**
   * @brief
   * @return
   */
  vid_t get_major_component_vertices() const {
    return major_component_vertices_;
  }
private:
  /**
   * @brief
   */
  void make_samples();

  /**
   * @brief
   * @param i
   */
  void print_progress(size_t i);

  /**
   * @brief
   */
  void compute_closeness_inv();

private:
  dualmode_engine_t<INCOMING, OUTGOING> *engine_;
  graph_info_t graph_info_;
  vid_t num_samples_;
  std::vector<vid_t> samples_;
  closeness_state_t closeness_;
  double fac_;

  connected_component_t<INCOMING, OUTGOING> *cc_;
  vid_t major_component_label_;
  vid_t major_component_vertices_;
};

template <typename INCOMING, typename OUTGOING>
david_closeness_t<INCOMING, OUTGOING>::david_closeness_t(dualmode_engine_t<INCOMING, OUTGOING> *engine,
    const graph_info_t& graph_info, const david_opts_t& opts) : engine_(engine), 
    graph_info_(graph_info),
    closeness_(graph_info.max_v_i_, engine->in_edges()->partitioner()) {
  cc_ = new connected_component_t<INCOMING, OUTGOING>(engine, graph_info);
  cc_->compute();
  
  major_component_label_ = cc_->get_major_label();
  major_component_vertices_ = cc_->get_major_vertices();
  LOG(INFO) << "major_label: " << major_component_label_ << " major_vertices: " << major_component_vertices_ << std::endl;

  num_samples_ = std::min(opts.num_samples_, major_component_vertices_);
  fac_ = 1.0 * major_component_vertices_ / num_samples_ /
         (major_component_vertices_ - 1);

  samples_.resize(num_samples_);
  closeness_.fill(0.0f);

}

template <typename INCOMING, typename OUTGOING>
david_closeness_t<INCOMING, OUTGOING>::~david_closeness_t() {
  delete cc_;
}

// partition 0 randomly select num_samples samples and broadcast them to other
// partitions
template <typename INCOMING, typename OUTGOING>
void david_closeness_t<INCOMING, OUTGOING>::make_samples() {
  auto p = cc_->get_labels();
  auto active_all = engine_->alloc_v_subset();
  active_all.fill();
  auto active_view_all = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), active_all);
  auto global_label = engine_->template alloc_v_state<vid_t>();
  active_view_all.template foreach<vid_t>([&](vid_t v_i){
      global_label[v_i] = (*p)[v_i];
    return 0;
  });

  auto& cluster_info = plato::cluster_info_t::get_instance();
  if (cluster_info.partition_id_ != 0) {
    vid_t v_begin = engine_->out_edges()->partitioner()->offset_[cluster_info.partition_id_];
    vid_t v_end = engine_->out_edges()->partitioner()->offset_[cluster_info.partition_id_+1]; 
    MPI_Send(&global_label[v_begin], v_end - v_begin, get_mpi_data_type<vid_t>(), 0, 0, MPI_COMM_WORLD);
  }
  else {
    for (int i = 1; i < cluster_info.partitions_; ++i ) {
      MPI_Status recv_status;
      vid_t v_begin = engine_->out_edges()->partitioner()->offset_[i];
      vid_t v_end = engine_->out_edges()->partitioner()->offset_[i+1]; 
      MPI_Recv(&global_label[v_begin], v_end - v_begin, get_mpi_data_type<vid_t>(), i, 0, MPI_COMM_WORLD, &recv_status);
    }

    //for (vid_t i = 0; i < graph_info_.vertices_; ++i) {
    //  LOG(INFO) << "vertex: " << i << " label: " << global_label[i] << std::endl;
    //}
  }

  if (cluster_info.partition_id_ == 0) {
    std::unordered_set<vid_t> unique_samples;
    while (unique_samples.size() < num_samples_) {
      vid_t r = std::rand() % graph_info_.vertices_;
      if (global_label[r] != major_component_label_) continue;
      unique_samples.insert(r);
    }
    std::copy(unique_samples.begin(), unique_samples.end(), samples_.begin());
  }
  MPI_Bcast(&samples_[0], num_samples_, get_mpi_data_type<vid_t>(), 0, MPI_COMM_WORLD);

  LOG(INFO) << "make samples finish  num_samples: " << num_samples_ << std::endl;
}

template <typename INCOMING, typename OUTGOING>
void david_closeness_t<INCOMING, OUTGOING>::print_progress(size_t i) {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  LOG_IF(INFO, samples_.size() >= 50 && i % (samples_.size() / 50) == 0 && cluster_info.partition_id_ == 0)
    << "process " << i << " of " << samples_.size() << " samples ("
    << i * 100 / samples_.size() << "%)\n";
}

template <typename INCOMING, typename OUTGOING>
void david_closeness_t<INCOMING, OUTGOING>::compute_closeness_inv() {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  vid_t v_begin = engine_->out_edges()->partitioner()->offset_[cluster_info.partition_id_];
  vid_t v_end = engine_->out_edges()->partitioner()->offset_[cluster_info.partition_id_+1]; 
#pragma omp parallel for
  for (vid_t i = v_begin; i < v_end; i++) {
    closeness_[i] = 1.0 / closeness_[i];
    if (std::isinf(closeness_[i])) {
      closeness_[i] = 0;
    }
  }
}

template <typename INCOMING, typename OUTGOING>
void david_closeness_t<INCOMING, OUTGOING>::compute() {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  make_samples();

  auto accu_func = [&](vid_t v_i, vid_t * val) {
    closeness_[v_i] += *val * fac_; return *val;
  };

  double exe_time = 0;
  exe_time -= MPI_Wtime();
  for (size_t i = 0; i < samples_.size(); i++) {
    vid_t root = samples_[i];
    calc_distance<INCOMING, OUTGOING>(engine_, root, accu_func);
    print_progress(i);
  }

  // the closeness_ is the reverse
  compute_closeness_inv();

  exe_time += MPI_Wtime();

  LOG_IF(INFO, cluster_info.partition_id_ == 0)
      << "performance: " << samples_.size() / exe_time << " vps\n";
}


template <typename INCOMING, typename OUTGOING>
double david_closeness_t<INCOMING, OUTGOING>::get_closeness_of(vid_t v) const {
  double cl = 0;
  double target_id = engine_->out_edges()->partitioner()->get_partition_id(v);

  auto& cluster_info = plato::cluster_info_t::get_instance();
  if (cluster_info.partition_id_ == target_id) {
    cl = closeness_[v];
  }

  MPI_Bcast(&cl, 1, MPI_DOUBLE, target_id, MPI_COMM_WORLD);
  return cl;
}

template <typename INCOMING, typename OUTGOING>
template <typename STREAM>
void david_closeness_t<INCOMING, OUTGOING>::save(std::vector<STREAM*>& ss) {
  boost::lockfree::queue<david_msg_type_t> que(1024);
  LOG_IF(FATAL, !que.is_lock_free())
    << "boost::lockfree::queue is not lock free\n";

  // start a thread to pop and edge and write to output
  std::atomic<bool> done(false);
  std::thread pop_write([&done, &ss, &que](void) {
#pragma omp parallel num_threads(ss.size())
    {
      int tid = omp_get_thread_num();
      david_msg_type_t vb;
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

  auto active_all = engine_->alloc_v_subset();
  active_all.fill();
  //traverse
  auto active_view_all = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), active_all);
  active_view_all.template foreach<vid_t>([&](vid_t v_i){
    while (!que.push(david_msg_type_t {v_i, closeness_[v_i] })) {
    }
    return 1;
  });

  done = true;
  pop_write.join();

}



}  // namespace algo
}  // namespace plato
#endif
