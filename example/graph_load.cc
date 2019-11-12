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

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "plato/util/perf.hpp"
#include "plato/graph/structure.hpp"

DEFINE_string(input,      "",    "input file, in csv format, without edge data");
DEFINE_bool(is_directed,  false, "is graph directed or not");
DEFINE_int32(alpha,       -1,    "alpha value used in sequence balance partition");

bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

DEFINE_validator(input, &string_not_empty);

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

int main(int argc, char** argv) {
  init(argc, argv);
  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.initialize(&argc, &argv);

  plato::stop_watch_t watch;

  watch.mark("t0");
  watch.mark("t1");

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = FLAGS_is_directed;
  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, FLAGS_input,
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << graph_info.edges_;
    LOG(INFO) << "vertices:     " << graph_info.vertices_;
    LOG(INFO) << "max_v_id:     " << graph_info.max_v_i_;
    LOG(INFO) << "is_directed_: " << graph_info.is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::vector<uint32_t> degrees = plato::generate_dense_out_degrees<uint32_t>(graph_info, *cache);
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "generate_dense_out_degrees cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  plato::eid_t __edges = graph_info.edges_;
  if (false == graph_info.is_directed_) { __edges = __edges * 2; }

  std::shared_ptr<plato::sequence_balanced_by_destination_t> part_bcsr(
      new plato::sequence_balanced_by_destination_t(degrees.data(),
        graph_info.vertices_, __edges, FLAGS_alpha)
  );
  part_bcsr->check_consistency();

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "partition cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t> bcsr(part_bcsr);
  CHECK(0 == bcsr.load_from_cache(graph_info, *cache));

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build bcsr cost: " << watch.show("t1") / 1000.0 << "s";
  }

  plato::mem_status_t mstatus;

  plato::self_mem_usage(&mstatus);
  LOG(INFO) << "memory usage(bcsr + cache): " << (double)mstatus.vm_rss / 1024.0 << " MBytes";

  cache = nullptr;  // destroy edges cache

  plato::self_mem_usage(&mstatus);
  LOG(INFO) << "memory usage(bcsr): " << (double)mstatus.vm_rss / 1024.0 << " MBytes";

  std::shared_ptr<plato::sequence_balanced_by_source_t> part_dcsc (
      new plato::sequence_balanced_by_source_t(part_bcsr->offset_)
  );

  watch.mark("t1");

  plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t> dcsc(part_dcsc);

  {
    plato::traverse_opts_t opts; opts.mode_ = plato::traverse_mode_t::RANDOM;
    CHECK(0 == dcsc.load_from_graph(graph_info, bcsr, true, opts));
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build dcsc cost: " << watch.show("t1") / 1000.0 << "s";
    LOG(INFO) << "total cost: " << watch.show("t0") / 1000.0 << "s";
  }

  plato::self_mem_usage(&mstatus);
  LOG(INFO) << "memory usage(bcsr + dcsc): " << (double)mstatus.vm_rss / 1024.0 << " MBytes";

  return 0;
}

