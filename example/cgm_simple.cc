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

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/graph/graph.hpp"
#include "plato/algo/cgm/connected_component.hpp"

DEFINE_string(input,         "",     "input file, in csv format, without edge data");
DEFINE_string(output,        "",     "output directory, store the closeness result");
DEFINE_bool(is_directed,     false,  "is graph directed or not");
DEFINE_int32(alpha,          -1,     "alpha value used in sequence balance partition");
DEFINE_int32(label,          -1,     "");
DEFINE_bool(part_by_in,      false,  "partition by in-degree");
DEFINE_string(output_method, "sub_graph_by_label",     "");
DEFINE_string(vtype,         "uint32",                 "");
DEFINE_bool(need_encode,     false,                    "");

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

template <typename VID_T>
void run_cgm(bool need_encode) {
  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  plato::vid_encoder_t<plato::empty_t, VID_T> data_encoder;

  auto encoder_ptr = &data_encoder;
  if (!need_encode) encoder_ptr = nullptr;

  plato::graph_info_t graph_info(FLAGS_is_directed);
  auto graph = plato::create_dualmode_seq_from_path<plato::empty_t, VID_T>(&graph_info, FLAGS_input,
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>,
      FLAGS_alpha, FLAGS_part_by_in, encoder_ptr);

  watch.mark("t0");
  using bcsr_spec_t = plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>;
  using dcsc_spec_t = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;

  plato::dualmode_engine_t<dcsc_spec_t, bcsr_spec_t> engine (
    std::shared_ptr<dcsc_spec_t>(&graph.second,  [](dcsc_spec_t*) { }),
    std::shared_ptr<bcsr_spec_t>(&graph.first, [](bcsr_spec_t*) { }),
    graph_info);

  plato::algo::connected_component_t<dcsc_spec_t, bcsr_spec_t> cc(&engine, graph_info);
  cc.compute();

  if (cluster_info.partition_id_ == 0) {
    LOG(INFO) << cc.template get_summary<VID_T>(encoder_ptr);
  }

  if (FLAGS_output_method == "sub_graph_by_label") {
    cc.template write_component<VID_T>(FLAGS_output, FLAGS_label, encoder_ptr);
  }
  else if (FLAGS_output_method == "all_vertices") {
    cc.template write_all_vertices<VID_T>(FLAGS_output, encoder_ptr);
  } else if (FLAGS_output_method == "all_edges") {
    cc.template write_all_edges<VID_T>(FLAGS_output, encoder_ptr);
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "connected component done const: " << watch.show("t0") / 1000.0 << "s";
  }
}

int main(int argc, char** argv) {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  LOG(INFO) << "partitions: " << cluster_info.partitions_ << " partition_id: " << cluster_info.partition_id_ << std::endl;

  bool need_encode = FLAGS_need_encode;

  if (FLAGS_vtype == "uint32")  {
    run_cgm<uint32_t>(need_encode);
  } else { 

    LOG(INFO) << "If vertex type is not uint32_t, It must use encoder";
    need_encode = true;

    if (FLAGS_vtype == "int32") {
      run_cgm<int32_t>(need_encode);
    } else if (FLAGS_vtype == "int64") {
      run_cgm<int64_t>(need_encode);
    } else if (FLAGS_vtype == "uint64") {
      run_cgm<uint64_t>(need_encode);
    }
  }

  return 0;
}

