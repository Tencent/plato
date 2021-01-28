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
#include "plato/algo/infomap/infomap.hpp"

DEFINE_string(input,        "",     "input file, in csv format");
DEFINE_string(output,       "",     "output file, in csv format");
DEFINE_bool(is_directed,    false,  "is graph directed or not");
DEFINE_int32(alpha,         -1,     "alpha value used in sequence balance partition");
DEFINE_bool(need_encode,    false,  "");
DEFINE_int32(pagerank_iter, 50,     "");
DEFINE_double(pagerank_threshold,   0.0001, "");
DEFINE_double(teleport_prob,        0.15,   "");
DEFINE_int32(inner_iter,    3,      "");
DEFINE_int32(outer_iter,    2,      "");
DEFINE_string(vtype,        "uint32",       "");
DEFINE_int32(comm_info_num, 100,            "");

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

template<typename VID_T>
void run_infomap(bool need_encode) {
  plato::stop_watch_t watch;

  plato::algo::infomap_opts_t opts;
  opts.input_ = FLAGS_input;
  opts.output_ = FLAGS_output;
  opts.alpha_ = FLAGS_alpha;
  opts.is_directed_ = FLAGS_is_directed;
  opts.need_encode_ = need_encode;
  opts.pagerank_iter_ = FLAGS_pagerank_iter;
  opts.pagerank_threshold_ = FLAGS_pagerank_threshold;
  opts.teleport_prob_ = FLAGS_teleport_prob;
  opts.inner_iter_ = FLAGS_inner_iter;
  opts.outer_iter_ = FLAGS_outer_iter;
  opts.comm_info_num_ = FLAGS_comm_info_num;

  plato::algo::infomap_t<VID_T> infomap(opts);
  watch.mark("t0");
  watch.mark("t1");
  infomap.compute();
  LOG(INFO) << "infomap compute cost: " << watch.show("t1") / 1000.0;
  infomap.statistic();
  watch.mark("t1");
  infomap.output();
  LOG(INFO) << "infomap output cost: " << watch.show("t1") / 1000.0;

  LOG(INFO) << "infomap total cost: " << watch.show("t0") / 1000.0;
}

int main(int argc, char** argv) {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  bool need_encode = FLAGS_need_encode;

  if (FLAGS_vtype == "uint32") {
    run_infomap<uint32_t>(need_encode);
  } else { 

    LOG(INFO) << "If vertex type is not uint32_t, It must use encoder";
    need_encode = true;

    if (FLAGS_vtype == "int32") {
      run_infomap<int32_t>(need_encode);
    } else if (FLAGS_vtype == "int64") {
      run_infomap<int64_t>(need_encode);
    } else if (FLAGS_vtype == "uint64") {
      run_infomap<uint64_t>(need_encode);
    }
  }
  return 0;
}

