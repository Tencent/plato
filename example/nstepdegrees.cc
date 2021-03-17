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
#include <vector>
#include <unordered_map>
#include <string>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/engine/dualmode.hpp"
#include "plato/graph/graph.hpp"
#include "plato/algo/nstepdegrees/nstepdegrees.hpp"

using namespace plato;

DEFINE_string(input,        "",     "input file, in csv format, without edge data");
DEFINE_string(output,       "",     "output directory, store the closeness result");
DEFINE_int32(alpha,        -1,      "alpha value used in sequence balance partition");
DEFINE_bool(part_by_in,    true,    "partition by in-degree");
DEFINE_bool(is_directed,   false,   "is graph directed or not");
DEFINE_int32(step,          -1,     "how many step's degree should be counted, -1 means infinity");
DEFINE_int32(bits,           6,      "hyperloglog bit width used for cardinality estimation");
DEFINE_string(type,        "both",  "count 'in' degree or 'out' degree or 'both'");
DEFINE_string(actives,     "",      "active vertex input in csv format, each line has one"
                                    " vertex id. if this parameter is given, nstepdegrees"
                                    " only calculate active vertex's nstepdegrees.");

/**
 * @brief not empty validator.
 * @param value
 * @return
 */
bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

DEFINE_validator(input,  &string_not_empty);
DEFINE_validator(output, &string_not_empty);
DEFINE_validator(actives, &string_not_empty);

using GRAPH_T     = std::pair<bcsr_t<plato::empty_t, sequence_balanced_by_destination_t>,dcsc_t<plato::empty_t, sequence_balanced_by_source_t>>;
using bcsr_spec_t = plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>;
using dcsc_spec_t = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;
using v_subset_t  = bitmap_t<>;

/**
 * @brief init
 * @param argc
 * @param argv
 */
void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

/**
 * @brief
 * @tparam BitWidth
 * @param engine
 * @param graph_info
 * @param graph
 * @param actives_v
 * @param opts
 */
template <uint32_t BitWidth>
void work_flow(dualmode_engine_t<dcsc_spec_t, bcsr_spec_t>* engine, const graph_info_t& graph_info, GRAPH_T& graph, const v_subset_t& actives_v, const plato::algo::nstepdegree_opts_t& opts) {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::algo::nstepdegrees_t<dcsc_spec_t, bcsr_spec_t, BitWidth> nstepdegrees(engine, graph_info, actives_v, opts);
  nstepdegrees.compute(graph.second, graph.first);

  plato::fs_mt_omp_output_t os(FLAGS_output, (boost::format("%04d_") % cluster_info.partition_id_).str(), false);
  nstepdegrees.save(os);

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

  plato::algo::nstepdegree_opts_t opts;
  opts.step = FLAGS_step;
  opts.type = FLAGS_type;
  opts.is_directed = FLAGS_is_directed;
  watch.mark("t0");

  plato::dualmode_engine_t<dcsc_spec_t, bcsr_spec_t> engine (
    std::shared_ptr<dcsc_spec_t>(&graph.second,  [](dcsc_spec_t*) { }),
    std::shared_ptr<bcsr_spec_t>(&graph.first, [](bcsr_spec_t*) { }),
    graph_info);
  
  auto actives_v = engine.alloc_v_subset();
  if(FLAGS_actives != "ALL") {
    plato::load_vertices_state_from_path<uint32_t>(FLAGS_actives, 
    plato::edge_format_t::CSV, graph.second.partitioner(), plato::uint32_t_decoder, 
    [&](plato::vertex_unit_t<uint32_t>&& unit) {
      actives_v.set_bit(unit.vid_);
    });
  } else {
    actives_v.fill();
  }
  
  switch (FLAGS_bits) {
    case 6:
      work_flow<6>(&engine, graph_info, graph, actives_v, opts);
      break;
    
    case 7:
      work_flow<7>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 8:
      work_flow<8>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 9:
      work_flow<9>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 10:
      work_flow<10>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 11:
      work_flow<11>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 12:
      work_flow<12>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 13:
      work_flow<13>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 14:
      work_flow<14>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 15:
      work_flow<15>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 16:
      work_flow<16>(&engine, graph_info, graph, actives_v, opts);
      break;
      
    default:
      CHECK(false) << "unsupport hyperloglog bit width: " << FLAGS_bits
        << ", supported range is in [6, 16]";
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "nstepdegrees done const: " << watch.show("t0") / 1000.0 << "s";
  }

  return 0;
}

