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

#include <cstdint>
#include <cstdlib>
#include <utility>

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"

DEFINE_string(input,       "",     "input file, in csv format, without edge data");
DEFINE_bool(is_directed,   false,  "is graph directed or not");
DEFINE_uint32(root,        0,      "start bfs from which vertex");
DEFINE_int32(alpha,        -1,     "alpha value used in sequence balance partition");
DEFINE_bool(part_by_in,    false,  "partition by in-degree");
DEFINE_uint32(type,        0,      "0 -- always pull, 1 -- push-pull, else -- push");

bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

int main(int argc, char** argv) {
  using bcsr_spec_t          = plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>;
  using dcsc_spec_t          = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;
  using partition_bcsr_t     = bcsr_spec_t::partition_t;
  using state_parent_t       = plato::dense_state_t<plato::vid_t, partition_bcsr_t>;
  using bitmap_spec_t        = plato::bitmap_t<>;
  using adj_unit_list_spec_t = bcsr_spec_t::adj_unit_list_spec_t;

  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  watch.mark("t0");

  plato::graph_info_t graph_info(FLAGS_is_directed);
  auto graph = plato::create_dualmode_seq_from_path<plato::empty_t>(&graph_info, FLAGS_input,
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>,
      FLAGS_alpha, FLAGS_part_by_in);

  watch.mark("t1");
  auto out_degrees = plato::generate_dense_out_degrees_fg<plato::vid_t>(graph_info, graph.first, true);
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "generate out-degrees from graph cost: " << watch.show("t1") / 1000.0 << "s";
  }

  plato::eid_t edges = graph_info.edges_;
  if (false == graph_info.is_directed_) { edges = edges * 2; }

  plato::vid_t actives = 1;
  state_parent_t parent(graph_info.max_v_i_, graph.first.partitioner());

  bitmap_spec_t visited(graph_info.vertices_);
  std::shared_ptr<bitmap_spec_t> active_current(new bitmap_spec_t(graph_info.vertices_));
  std::shared_ptr<bitmap_spec_t> active_next(new bitmap_spec_t(graph_info.vertices_));

  visited.set_bit(FLAGS_root);
  active_current->set_bit(FLAGS_root);
  parent.fill(graph_info.vertices_);
  parent[FLAGS_root] = FLAGS_root;

  auto partition_view = graph.first.partitioner()->self_v_view();

  watch.mark("t1");
  watch.mark("t2");

  bool is_sparse = true;
  plato::eid_t active_edges = 0;
  plato::vid_t last_actives = 0;
  for (int epoch_i = 0; 0 != actives; ++epoch_i) {
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "active_v[" << epoch_i << "] = " << last_actives
        << ", active_e[" << active_edges << "/" << is_sparse << "], cost: " << watch.show("t1") / 1000.0 << "s";;
      last_actives = actives;
    }

    watch.mark("t1");
    auto active_view = plato::create_active_v_view(partition_view, *active_current);

    {  // count active edges
      active_edges = 0;
      if (1 == FLAGS_type) {
        out_degrees.reset_traversal(active_current);
        #pragma omp parallel reduction(+:active_edges)
        {
          size_t chunk_size = 4 * PAGESIZE;
          plato::eid_t __active_edges = 0;

          while (out_degrees.next_chunk([&](plato::vid_t v_i, plato::vid_t* degrees) {
            __active_edges += (*degrees); return true;
          }, &chunk_size)) { }
          active_edges += __active_edges;
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, &active_edges, 1, plato::get_mpi_data_type<plato::eid_t>(), MPI_SUM, MPI_COMM_WORLD);

      is_sparse = (active_edges < edges / 20);  // switch between aggregate_message and spread_message
    }

    active_next->clear();
    if (0 == FLAGS_type || false == is_sparse)  {  // pull
      using context_spec_t = plato::mepa_ag_context_t<plato::vid_t>;
      using message_spec_t = plato::mepa_ag_message_t<plato::vid_t>;

      watch.mark("t11");
      visited.sync();

      if (0 == cluster_info.partition_id_) {
        LOG(INFO) << "bitmap sync time: " << watch.show("t11") / 1000.0 << "s";
      }

      plato::bsp_opts_t opts;
      opts.local_capacity_ = 32 * PAGESIZE;

      actives = plato::aggregate_message<plato::vid_t, int, dcsc_spec_t> (graph.second,
        [&](const context_spec_t& context, plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
          if (visited.get_bit(v_i)) { return ; }
          for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
            plato::vid_t src = it->neighbour_;
            if (active_current->get_bit(src)) {
              context.send(message_spec_t { v_i, src });
              break;
            }
          }
        },
        [&](int /*p_i*/, message_spec_t& message) {
          if (plato::cas(&parent[message.v_i_], graph_info.vertices_, message.message_)) {
            active_next->set_bit(message.v_i_);
            visited.set_bit(message.v_i_);
            return 1;
          }
          return 0;
        },
        opts
      );
    } else {  // push
      plato::bc_opts_t opts;
      opts.local_capacity_ = 4 * PAGESIZE;

      actives = plato::broadcast_message<plato::vid_t, plato::vid_t> (active_view,
        [&](const plato::mepa_bc_context_t<plato::vid_t>& context, plato::vid_t v_i) {
          context.send(v_i);
        },
        [&](int /* p_i */, const plato::vid_t& v_i) {
          plato::vid_t activated = 0;

          auto neighbours = graph.first.neighbours(v_i);
          for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
            plato::vid_t dst = it->neighbour_;
            if (
              (parent[dst] == graph_info.vertices_)
                &&
              (plato::cas(&parent[dst], graph_info.vertices_, v_i))
            ) {
              active_next->set_bit(dst);
              visited.set_bit(dst);
              ++activated;
            }
          }
          return activated;
        }, opts);
    }

    std::swap(active_next, active_current);
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "bfs cost: " << watch.show("t2") / 1000.0 << "s";
  }

  visited.sync();

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "found vertices: " << visited.count()
      << ", total cost: " << watch.show("t0") / 1000.0 << "s";
  }

  return 0;
}

