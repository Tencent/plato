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
#include <unordered_set>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/graph/graph.hpp"
#include "plato/algo/hanp/hanp.hpp"
#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/util/spinlock.hpp"

DEFINE_string(input,       "",      "input file, in csv format, without edge data");
DEFINE_string(output,      "",      "output directory");
DEFINE_bool(is_directed,   true,    "is graph directed or not");
DEFINE_bool(part_by_in,    true,    "partition by in-degree");
DEFINE_int32(alpha,        -1,      "alpha value used in sequence balance partition");
DEFINE_uint32(iterations,  20,      "number of iterations");
DEFINE_double(preference,  1.0,     "is any arbitrary comparable characteristic for any node");
DEFINE_double(hop_att,     0.1,     "a new attenuated score");

bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

DEFINE_validator(input,  &string_not_empty);
DEFINE_validator(output, &string_not_empty);

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

void print_flags(){
  LOG(INFO) << "input        : " << FLAGS_input;
  LOG(INFO) << "output       : " << FLAGS_output;
  LOG(INFO) << "is_directed     : " << FLAGS_is_directed;
  LOG(INFO) << "part_by_in   : " << FLAGS_part_by_in;
  LOG(INFO) << "alpha        : " << FLAGS_alpha;
  LOG(INFO) << "iterations   : " << FLAGS_iterations;
  LOG(INFO) << "preference   : " << FLAGS_preference;
  LOG(INFO) << "hot_att      : " << FLAGS_hop_att;
}

int main(int argc, char** argv) {

  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  if (0 == cluster_info.partition_id_) {
    print_flags();
  }
  watch.mark("t0");

  plato::graph_info_t graph_info(FLAGS_is_directed);

  auto pdcsc = plato::create_dcsc_seqd_from_path<float>(
    &graph_info, FLAGS_input, plato::edge_format_t::CSV,
    plato::float_decoder, FLAGS_alpha, FLAGS_part_by_in
  );

  using graph_spec_t = std::remove_reference<decltype(*pdcsc)>::type;

  plato::algo::hanp_opts_t opts;
  opts.iteration_ = FLAGS_iterations;
  opts.preference = FLAGS_preference;
  opts.hop_att    = FLAGS_hop_att;
  opts.dis        = 1e-5;

  auto labels = plato::algo::hanp<graph_spec_t>(*pdcsc, graph_info, opts);
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "hanp calculation done: " << watch.show("t0") / 1000.0 << "s";
  }

  watch.mark("t0");
  {
    // save result
    plato::thread_local_fs_output os(FLAGS_output, (boost::format("%04d_") % cluster_info.partition_id_).str(), true);

    labels.template foreach<int> (
      [&](plato::vid_t v_i, plato::vid_t* pval) {
        auto& fs_output = os.local();
        fs_output << v_i << "," << *pval << "\n";
        return 0;
      }
    );
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "save result cost: " << watch.show("t1") / 1000.0 << "s";
  }

  return 0;
}

