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
#include <string>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "plato/graph/graph.hpp"
#include "plato/algo/tree_stat/tree_stat.hpp"

DEFINE_string(input,       "",     "input file, in csv format, without edge data");
DEFINE_string(output,      "",     "result file");

DEFINE_uint64(vertices,   0,  "vertex count");
DEFINE_uint64(edges,      0,  "edge count");

DEFINE_bool(is_directed,   false,  "is graph directed or not");
DEFINE_uint32(root,        0,      "start bfs from which vertex");
DEFINE_int32(alpha,        -1,     "alpha value used in sequence balance partition");
DEFINE_bool(part_by_in,    false,  "partition by in-degree");

DEFINE_string(stat,      "",  "which status should compute, we support 'width', 'depth' now. "
                              " use ',' to separate multiple stat, eg: 'width,depth'");
DEFINE_bool(validate, false,  "strict validate tree's structure");

bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

DEFINE_validator(input, &string_not_empty);


void save_result(const plato::algo::tree_stat_t& stat,const std::string &value) {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  
  std::vector<std::string> splits;
  boost::split(splits, value, boost::is_any_of(","));
  plato::thread_local_fs_output os(FLAGS_output, (boost::format("%04d_") % cluster_info.partition_id_).str(), true);
  auto& fs_output = os.local();

  for (size_t i = 0; i < splits.size(); ++i) {
    if (0 != i) { fs_output << ","; }
    if ("width" == splits[i]) {
      fs_output << std::to_string(stat.width_);
    }
    if ("depth" == splits[i]) {
      fs_output << std::to_string(stat.depth_);
    }
  }
  fs_output << std::string("\n");
}

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
  auto graph = plato::create_dualmode_seq_from_path<plato::empty_t>(&graph_info, FLAGS_input,
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>,
      FLAGS_alpha, FLAGS_part_by_in);

  plato::vid_t root = FLAGS_root;

  watch.mark("t0");
  auto stat = plato::algo::tree_stat(graph.second, graph.first,
      graph_info, root);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "tree_stat done, w: "  << stat.width_
      <<"d: " << stat.depth_ << ", cost: " << " is tree :" << stat.is_tree_ << watch.show("t0") / 1000.0 << "s";

    save_result(stat,FLAGS_stat);
  }

  return 0;
}
