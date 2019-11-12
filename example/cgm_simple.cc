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

DEFINE_string(input,       "",     "input file, in csv format, without edge data");
DEFINE_string(output,       "",     "output directory, store the closeness result");
DEFINE_bool(is_directed,   false,  "is graph directed or not");
DEFINE_int32(alpha,        -1,     "alpha value used in sequence balance partition");
DEFINE_int32(label,        -1,     "");
DEFINE_bool(part_by_in,    false,  "partition by in-degree");

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

  watch.mark("t0");
  using bcsr_spec_t = plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>;
  using dcsc_spec_t = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;

  plato::dualmode_engine_t<dcsc_spec_t, bcsr_spec_t> engine (
    std::shared_ptr<dcsc_spec_t>(&graph.second,  [](dcsc_spec_t*) { }),
    std::shared_ptr<bcsr_spec_t>(&graph.first, [](bcsr_spec_t*) { }),
    graph_info);

  plato::algo::connected_component_t<dcsc_spec_t, bcsr_spec_t> cc(&engine, graph_info);
  cc.compute();
  LOG(INFO) << cc.get_summary();
  std::vector<plato::hdfs_t::fstream *> fsms;
  using cgm_stream_t = boost::iostreams::filtering_stream<boost::iostreams::output>;
  std::vector<cgm_stream_t*> fouts;
  for (int tid = 0; tid < cluster_info.threads_; ++tid) {
    char fn[FILENAME_MAX];
    sprintf(fn, "%s/part-%05d.csv.gz", FLAGS_output.c_str(),
        (cluster_info.partition_id_ * cluster_info.threads_ + tid));

    plato::hdfs_t &fs = plato::hdfs_t::get_hdfs(fn);
    fsms.emplace_back(new plato::hdfs_t::fstream(fs, fn, true));
    fouts.emplace_back(new cgm_stream_t());
    fouts.back()->push(boost::iostreams::gzip_compressor());
    fouts.back()->push(*fsms.back());
    LOG(INFO) << fn;
  }

  cc.write_component(fouts, FLAGS_label);

  for (int i = 0; i < cluster_info.threads_; ++i) {
    delete fouts[i];
    delete fsms[i];
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "connected component done const: " << watch.show("t0") / 1000.0 << "s";
  }

  return 0;
}

