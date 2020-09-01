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
#include <unordered_map>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "plato/graph/graph.hpp"
#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"

DEFINE_string(input,         "",     "input file, in csv format, without edge data");
DEFINE_bool(is_directed,     false,  "is graph directed or not");
DEFINE_int32(alpha,          -1,     "alpha value used in sequence balance partition");
DEFINE_bool(part_by_in,      false,  "partition by in-degree");
DEFINE_bool(src_need_encode, true,   "");
DEFINE_bool(dst_need_encode, true,   "");

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

int main(int argc, char** argv) {
  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();
  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  plato::graph_info_t graph_info(FLAGS_is_directed);
  plato::vid_encoder_opts_t opts;
  opts.src_need_encode_ = FLAGS_src_need_encode;
  opts.dst_need_encode_ = FLAGS_dst_need_encode;
  using EDATA = plato::empty_t;
  using VID_T = uint64_t;

  auto decoder = plato::dummy_decoder<plato::empty_t>;
  //auto decoder = plato::uint8_t_decoder;
  plato::vid_encoder_t<EDATA, VID_T, plato::edge_cache_t> data_encoder(opts);
  //auto graph = plato::create_bcsr_seqs_from_path<EDATA, VID_T>(&graph_info, 
  //  FLAGS_input, plato::edge_format_t::CSV, decoder,
  //  FLAGS_alpha, FLAGS_part_by_in, &data_encoder);

  auto tcsr = plato::create_tcsr_hashs_from_path<EDATA, plato::vid_t, std::hash<plato::vid_t>,
       VID_T, plato::edge_cache_t>(&graph_info, FLAGS_input, plato::edge_format_t::CSV,
       decoder, &data_encoder);

  auto data = data_encoder.data();

  LOG(INFO) << "data size: " << data.size();

  size_t stride = data.size() / 100;
  if (stride == 0) stride = 1;
  for (size_t i = 0; i < data.size(); i += stride) {
    LOG(INFO) << "pid: " << cluster_info.partition_id_ << " i: " << i << " val: " << data[i];
  }
  return 0;
}

