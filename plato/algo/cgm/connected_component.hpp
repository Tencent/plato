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

#ifndef __PLATO_ALGO_CGM_CONNECTED_COMPONENT_HPP__
#define __PLATO_ALGO_CGM_CONNECTED_COMPONENT_HPP__

#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <vector>
#include <boost/format.hpp>
#include <boost/lockfree/queue.hpp>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"

namespace plato { namespace algo {

/**
 * @brief
 * @tparam INCOMING
 * @tparam OUTGOING
 */
template <typename INCOMING, typename OUTGOING>
class connected_component_t {
public:
  /**
   * @brief
   */
  struct comp_info_t {
    vid_t vertices_;
    eid_t edges_;
    comp_info_t() : vertices_(0), edges_(0) {}
  };
  /**
   * @brief
   */
  struct connected_comp_msg_type_t {
    vid_t src_;
    vid_t label_;
  };

  using partition_t = typename dualmode_detail::partition_traits<INCOMING, OUTGOING>::type;
  using cgm_state_t = dense_state_t<vid_t, partition_t>;
  using edata_t = typename INCOMING::edata_t;

public:
  explicit connected_component_t(dualmode_engine_t<INCOMING, OUTGOING> * engine, const graph_info_t& graph_info);
  ~connected_component_t();

  /**
   * @brief run
   */
  void compute();

  /**
   * @brief
   * @tparam VID_T
   * @param vid_encoder
   * @return
   */
  template <typename VID_T>
  std::string get_summary(vencoder_t<edata_t, VID_T> vid_encoder = nullptr) const;

  /**
   * @brief
   * @tparam VID_T
   * @param prefix
   * @param label
   * @param vid_encoder
   */
  template <typename VID_T>
  void write_component(const std::string& prefix, VID_T label, vencoder_t<edata_t, VID_T> vid_encoder = nullptr);

  /**
   * @brief
   * @tparam VID_T
   * @param prefix
   * @param vid_encoder
   */
  template <typename VID_T>
  void write_all_vertices(const std::string& prefix, vencoder_t<edata_t, VID_T> vid_encoder = nullptr);


  /**
   * @brief
   * @tparam VID_T
   * @param prefix
   * @param vid_encoder
   */
  template <typename VID_T>
  void write_all_edges(const std::string& prefix, vencoder_t<edata_t, VID_T> vid_encoder = nullptr);

  /**
   * @brief
   * @return
   */
  int get_num_components() const { return global_label_info_.size(); }

  /**
   * @brief
   * @return
   */
  std::unordered_map<vid_t, comp_info_t>* get_components_info() {
    return &global_label_info_; 
  }

  /**
   * @brief major_label_ getter
   * @return
   */
  vid_t get_major_label() const { return major_label_; }

  /**
   * @brief
   * @return
   */
  vid_t get_major_vertices() const {
    const auto it = global_label_info_.find(major_label_);
    return it->second.vertices_;
  }

  /**
   * @brief
   * @return
   */
  cgm_state_t* get_labels() { return &labels_; }

private:
  std::unordered_map<vid_t, comp_info_t> global_label_info_;
  vid_t major_label_;
  dualmode_engine_t<INCOMING, OUTGOING> * engine_;
  graph_info_t graph_info_;
  cgm_state_t labels_;
};

template <typename INCOMING, typename OUTGOING>
connected_component_t<INCOMING, OUTGOING>::connected_component_t(
  dualmode_engine_t<INCOMING, OUTGOING> * engine,
  const graph_info_t& graph_info)
  : engine_(engine), graph_info_(graph_info),
    labels_(graph_info.max_v_i_, engine->in_edges()->partitioner()) {
}

template <typename INCOMING, typename OUTGOING>
connected_component_t<INCOMING, OUTGOING>::~connected_component_t() {
}

template <typename INCOMING, typename OUTGOING>
void connected_component_t<INCOMING, OUTGOING>::compute() {
  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  // alloc struct used during iteration
  auto active_current = engine_->alloc_v_subset();
  auto active_next = engine_->alloc_v_subset();

  // init structs
  active_current.fill();
  auto active_view = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), active_current);
  vid_t actives = active_view.template foreach<vid_t>([&](vid_t v_i) {
    labels_[v_i] = v_i;
    return 1;
  });

  //iteration
  using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;
  for (int epoch_i = 0; 0 != actives; ++epoch_i) {

    using pull_context_t = plato::template mepa_ag_context_t<connected_comp_msg_type_t>;
    using pull_message_t = plato::template mepa_ag_message_t<connected_comp_msg_type_t>;
    using push_context_t = plato::template mepa_bc_context_t<connected_comp_msg_type_t>;
    watch.mark("t1");
    active_next.clear();
    actives = engine_->template foreach_edges<connected_comp_msg_type_t, vid_t> (
      [&](const push_context_t& context, vid_t v_i) {
        context.send(connected_comp_msg_type_t{v_i, labels_[v_i]});
      },
      [&](int /*p_i*/, connected_comp_msg_type_t & msg) {
        vid_t activated = 0;

        auto neighbours = engine_->out_edges()->neighbours(msg.src_);
        for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
          vid_t dst = it->neighbour_;
          if (msg.label_ < labels_[dst]) {
            plato::write_min(&(labels_[dst]), msg.label_);
            active_next.set_bit(dst);
            ++activated;
          }
        }
        return activated;
      },
      [&](const pull_context_t& context, vid_t v_i, const adj_unit_list_spec_t& adjs) {
        vid_t cur_label = v_i; //mirror node
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          vid_t src = it->neighbour_;
          if (labels_[src] < cur_label) {
            cur_label = labels_[src];
          }
        }
        if (cur_label < v_i) {
          context.send(pull_message_t { v_i, connected_comp_msg_type_t{v_i, cur_label} });
        }
      },
      [&](int, pull_message_t& msg) {
        if (msg.message_.label_ < labels_[msg.v_i_]) {
          plato::write_min(&(labels_[msg.v_i_]), msg.message_.label_);
          active_next.set_bit(msg.v_i_);
          return 1;
        }
        return 0;
      },
      active_current
    );

    std::swap(active_current, active_next);

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "active_v[" << epoch_i << "] = " << actives << ", cost: " << watch.show("t1") / 1000.0 << "s";
    }
  }

  //calc global label info
  LOG(INFO) << "begin calc global label";
  int partitions = cluster_info.partitions_;
  int partition_id = cluster_info.partition_id_;
  std::unordered_map<vid_t, vid_t> label_cnt;
  auto active_all = engine_->alloc_v_subset();
  active_all.fill();

  plato::vid_t v_begin = engine_->out_edges()->partitioner()->offset_[partition_id];
  plato::vid_t v_end = engine_->out_edges()->partitioner()->offset_[partition_id+1];

  for (auto v_i = v_begin; v_i < v_end; ++v_i) {
    label_cnt[labels_[v_i]] ++;
  }

  std::vector<vid_t> label_vec;
  std::vector<vid_t> count_vec;
  std::transform(
    label_cnt.begin(), label_cnt.end(), std::back_inserter(label_vec),
    [](const std::pair<vid_t, vid_t> &p) { return p.first; });
  std::transform(
    label_cnt.begin(), label_cnt.end(), std::back_inserter(count_vec),
    [](const std::pair<vid_t, vid_t> &p) { return p.second; });

  //rank 0 gather all label label from other ranks
  auto allgatherv =
    [&partitions](const std::vector<vid_t> &vec) -> std::vector<vid_t> {
      std::vector<int> counts(partitions, 0);
      std::vector<int> displ(partitions, 0);

      int size = vec.size();
      MPI_Gather(&size, 1, MPI_INT, &counts[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

      for (size_t i = 1; i < counts.size(); i++) {
        displ[i] = displ[i - 1] + counts[i - 1];
      }

      int sum = std::accumulate(counts.begin(), counts.end(), 0);
      MPI_Bcast(&sum, 1, MPI_INT, 0, MPI_COMM_WORLD);

      std::vector<vid_t> all(sum, 0);
      MPI_Gatherv(
        &vec[0], vec.size(), get_mpi_data_type<vid_t>(), &all[0], &counts[0],
        &displ[0], get_mpi_data_type<vid_t>(), 0, MPI_COMM_WORLD);
      bcast(&all[0], all.size(), get_mpi_data_type<vid_t>(), 0, MPI_COMM_WORLD);
      return all;
    };

  // gather the # of label_vec from each process
  std::vector<vid_t> all_label = allgatherv(label_vec);
  std::vector<vid_t> all_count = allgatherv(count_vec);
  LOG(INFO) << "allgatherv finish";

  // accumulate vertices
  for (size_t i = 0; i < all_label.size(); i++) {
    global_label_info_[all_label[i]].vertices_ += all_count[i];
  }

  std::vector<eid_t> comp_edges(graph_info_.vertices_, 0);
  if (cluster_info.partition_id_ == 0) {
    LOG(INFO) << "there are " << global_label_info_.size() << " components";
  }

  auto traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
    for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
      vid_t src = it->neighbour_;
      vid_t id = labels_[src];
      plato::write_add(&comp_edges[id], (eid_t)1);
    }
    return true;
  };

  engine_->in_edges()->reset_traversal();
  #pragma omp parallel num_threads(cluster_info.threads_)
  {
    size_t chunk_size = 256;
    while (engine_->in_edges()->next_chunk(traversal, &chunk_size)) { }
  }

  LOG(INFO) << "calc edges in each components";
  allreduce(MPI_IN_PLACE, &comp_edges[0], comp_edges.size(), get_mpi_data_type<eid_t>(), MPI_SUM,
      MPI_COMM_WORLD);
  vid_t major_label_cnt = 0;
  for (auto &e : global_label_info_) {
    e.second.edges_ = comp_edges[e.first];
    if (!graph_info_.is_directed_) {
      e.second.edges_ /= 2;
    }
    if (e.second.vertices_ > major_label_cnt) {
      major_label_cnt = e.second.vertices_;
      major_label_ = e.first;
    }
  }

}

template <typename INCOMING, typename OUTGOING>
template <typename VID_T>
std::string connected_component_t<INCOMING, OUTGOING>::get_summary(
    vencoder_t<edata_t, VID_T> vid_encoder) const {
  std::stringstream ss;

  ss << "Connected component summary: \n";
  ss << "There are " << get_num_components() << " components.\n";

  /// print components according to the descending order of vertices
  struct triple{
    vid_t label;
    vid_t vertices;
    eid_t edges;

    triple(vid_t l, vid_t v, eid_t e)
      : label(l), vertices(v), edges(e) {}
  };

  std::vector<triple *> comps;
  std::transform(
    global_label_info_.begin(), global_label_info_.end(), std::back_inserter(comps),
    [](const std::pair<vid_t, comp_info_t> &p) {
      return new triple(p.first, p.second.vertices_, p.second.edges_);
    });

  std::sort(comps.begin(), comps.end(), [](const triple *a, const triple *b) {
    return a->vertices > b->vertices;
  });

  int count = 0;
  for (const auto &c : comps) {
    ss << "Component #"  << ++count;
    if (vid_encoder == nullptr) ss << ", label=" << c->label;
    else ss << ", label=" << vid_encoder->data()[c->label];

    ss << boost::format(", vertices=%u, edges=%lu\n") %
        c->vertices % c->edges;
  }

  std::for_each(comps.begin(), comps.end(), std::default_delete<triple>());

  return ss.str();
}

template <typename INCOMING, typename OUTGOING>
template <typename VID_T>
void connected_component_t<INCOMING, OUTGOING>::write_component(
  const std::string& prefix, VID_T target_label, vencoder_t<edata_t, VID_T> vid_encoder) {

  auto& cluster_info = plato::cluster_info_t::get_instance();
  vid_t actual_label = graph_info_.vertices_;
  if (vid_encoder != nullptr && target_label != (VID_T)-1) {
    //need trans to encoded id
    vid_t tmp = graph_info_.vertices_;
    #pragma omp parallel num_threads(cluster_info.threads_)
    for (vid_t i = 0; i < graph_info_.vertices_; ++i) {
      if (vid_encoder->data()[i] == target_label) {
        tmp = i; 
      }
    }
    actual_label = tmp;
  } else {
    if (target_label == (VID_T)-1) {
      actual_label = major_label_;
    } else {
      actual_label = (vid_t)target_label;
    }
  }

  if (actual_label >= graph_info_.vertices_) {
    return;
  }

  thread_local_fs_output os(prefix, (boost::format("part_%04d_") % cluster_info.partition_id_).str(), true);

  if (actual_label != (vid_t)-1 && 
      global_label_info_.find(actual_label) == global_label_info_.end()) {
    //not a component id, regard as vertex id and try to find its component id  
    vid_t tmp = graph_info_.vertices_;
    if (engine_->in_edges()->partitioner()->get_partition_id(actual_label)
        == cluster_info.partition_id_) {
      tmp = labels_[actual_label];
    }
    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, get_mpi_data_type<vid_t>(), MPI_MIN, MPI_COMM_WORLD);
    actual_label = tmp;
  }

  auto output_result = [&](vid_t src, vid_t dst) {
    auto& fs_output = os.local();
    if (vid_encoder != nullptr) {
      fs_output << vid_encoder->decode(src) << "," << vid_encoder->decode(dst) << "\n";
    } else {
      fs_output << src << "," << dst << "\n";
    }
  };

  using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;
  auto traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
    for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
      vid_t src = it->neighbour_;
      vid_t src_label = labels_[src];
      if (src_label == actual_label) {
        if (!graph_info_.is_directed_) {
          if (src < v_i) {
            output_result(src, v_i);
          } 
        }
        else {
          output_result(src, v_i);
        }
      }
    }
    return true;
  };

  engine_->in_edges()->reset_traversal();
  #pragma omp parallel num_threads(cluster_info.threads_)
  {
    size_t chunk_size = 256;
    while (engine_->in_edges()->next_chunk(traversal, &chunk_size)) { }
  }

}

template <typename INCOMING, typename OUTGOING>
template <typename VID_T>
void connected_component_t<INCOMING, OUTGOING>::write_all_vertices(const std::string& prefix, 
    vencoder_t<edata_t, VID_T> vid_encoder) {

  auto actives = engine_->alloc_v_subset();
  actives.fill();
  auto active_view = plato::create_active_v_view(engine_->out_edges()->partitioner()->self_v_view(), actives);

  auto& cluster_info = plato::cluster_info_t::get_instance();
  thread_local_fs_output os(prefix, (boost::format("part_%04d_") % cluster_info.partition_id_).str(), true);

  auto output_result = [&](vid_t v) {
    auto& fs_output = os.local();
    if (vid_encoder != nullptr) {
      fs_output << vid_encoder->decode(v) << "," << vid_encoder->decode(labels_[v]) << "\n";
    } else {
      fs_output << v << "," << labels_[v] << "\n";
    }
  };

  active_view.template foreach<vid_t>([&](vid_t v_i) {
    output_result(v_i);
    return 0;
  });

}

template <typename INCOMING, typename OUTGOING>
template <typename VID_T>
void connected_component_t<INCOMING, OUTGOING>::write_all_edges(const std::string& prefix, 
    vencoder_t<edata_t, VID_T> vid_encoder) {

  auto& cluster_info = plato::cluster_info_t::get_instance();
  thread_local_fs_output os(prefix, (boost::format("part_%04d_") % cluster_info.partition_id_).str(), true);

  auto output_result = [&](vid_t label, vid_t src, vid_t dst) {
    auto& fs_output = os.local();
    if (vid_encoder != nullptr) {
      fs_output << vid_encoder->decode(label) << "," << 
          vid_encoder->decode(src) << "," << vid_encoder->decode(dst) << "\n";
    } else {
      fs_output << label << "," << src << "," << dst << "\n";
    }
  };

  using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;
  auto traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
    for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
      vid_t src = it->neighbour_;
      vid_t src_label = labels_[src];
      if (!graph_info_.is_directed_) {
        if (src < v_i) {
          output_result(src_label, src, v_i);
        } 
      }
      else {
        output_result(src_label, src, v_i);
      }
      
    }
    return true;
  };

  engine_->in_edges()->reset_traversal();
  #pragma omp parallel num_threads(cluster_info.threads_)
  {
    size_t chunk_size = 256;
    while (engine_->in_edges()->next_chunk(traversal, &chunk_size)) { }
  }
}


}  // namespace plato
}  // namespace algo
#endif
