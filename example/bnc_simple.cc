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
#include "plato/algo/bnc/betweenness.hpp"

DEFINE_string(input,       "",     "input file, in csv format, without edge data");
DEFINE_string(output,      "",     "output directory, store the closeness result");
DEFINE_bool(is_directed,   false,  "is graph directed or not");
DEFINE_int32(alpha,        -1,     "alpha value used in sequence balance partition");
DEFINE_bool(part_by_in,    false,  "partition by in-degree");
DEFINE_int32(chosen,       -1,     "chosen vertex");
DEFINE_int32(max_iteration, 0,     "");
DEFINE_double(constant,     2,     "");

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
  LOG(INFO) << "partitions: " << cluster_info.partitions_ << " partition_id: " << cluster_info.partition_id_ << std::endl;

  plato::graph_info_t graph_info(FLAGS_is_directed);
  auto graph = plato::create_dualmode_seq_from_path<plato::empty_t>(&graph_info, FLAGS_input,
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>,
      FLAGS_alpha, FLAGS_part_by_in);

  plato::algo::bader_opts_t opts;
  opts.chosen_ = FLAGS_chosen;
  opts.max_iteration_ = FLAGS_max_iteration;
  opts.constant_ = FLAGS_constant;
  watch.mark("t0");
  using bcsr_spec_t = plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>;
  using dcsc_spec_t = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;

  plato::dualmode_engine_t<dcsc_spec_t, bcsr_spec_t> engine (
    std::shared_ptr<dcsc_spec_t>(&graph.second,  [](dcsc_spec_t*) { }),
    std::shared_ptr<bcsr_spec_t>(&graph.first, [](bcsr_spec_t*) { }),
    graph_info);

  plato::algo::bader_betweenness_t<dcsc_spec_t, bcsr_spec_t, double> bader(&engine, graph_info, opts);
  bader.compute();

  plato::thread_local_fs_output os(FLAGS_output, (boost::format("%04d_") % cluster_info.partition_id_).str(), true);

  bader.save([&] (plato::vid_t v_i, double value) {
    auto& fs_output = os.local();
    fs_output << v_i << "," << value << "\n";
  });

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "bnc done const: " << watch.show("t0") / 1000.0 << "s";
  }

  return 0;
}

