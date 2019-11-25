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

#include <limits>
#include <memory>
#include <string>
#include <algorithm>

#include "boost/format.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/hdfs.hpp"
#include "plato/util/foutput.h"
#include "plato/graph/graph.hpp"
#include "plato/engine/walk.hpp"
#include "plato/graph/structure/hnbbcsr.hpp"

DEFINE_string(input,       "",     "input graph file, in csv format, eg: src,dst[,weight]");
DEFINE_string(types,       "",     "vertices type directory, format as 'vertex_id,type_id', \
                                    type_id should keep as small as possible");
DEFINE_string(metapath,    "",     "metapath restriction, eg: '0-1-2'");
DEFINE_string(output,      "",     "output path, in csv format, gzip compressed");
DEFINE_bool(is_directed,   false,  "is graph directed or not");
DEFINE_bool(is_weighted,   false,  "random walk with bias or not");
DEFINE_uint32(epoch,       5,      "how many epoch should perform");
DEFINE_uint32(step,        5,      "steps per epoch");
DEFINE_double(rate,        0.1,    "start 'rate'% walker per one bsp, reduce memory consumption");

static std::vector<uint32_t> _metapath;

/**
 * @brief validator.
 * @param value
 * @return
 */
static bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

/**
 * @brief
 * @param value
 * @return
 */
static bool metapath_validator(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }

  std::vector<std::string> splits;
  boost::split(splits, value, boost::is_any_of("-"));

  for (const auto& path: splits) {
    uint32_t id = (uint32_t)std::strtoul(path.c_str(), nullptr, 10);
    _metapath.emplace_back((uint8_t)id);
  }

  if (0 == _metapath.size()) {
    return false;
  }
  return true;
}

/**
 * @brief debug info.
 * @param notice
 */
void print_mem_info(std::string notice) {
  LOG(INFO) << "Notice: " << notice;
  plato::mem_status_t mstat;
  self_mem_usage(&mstat);
  LOG(INFO) << "Current memory usage : " << (double)mstat.vm_rss / 1000.0 << "MBytes";
  LOG(INFO) << "Current memory peak : " << (double)mstat.vm_peak / 1000.0 << "MBytes";
  LOG(INFO) << "Current resident set peak : " << (double)mstat.vm_hwm / 1000.0 << "MBytes";
}

/**
 * @brief
 * @tparam T
 * @param value
 * @return
 */
template <typename T>
static bool integer_not_zero(const char*, T value) {
  if (0 == value) { return false; }
  return true;
}

DEFINE_validator(input,    &string_not_empty);
DEFINE_validator(types,    &string_not_empty);
DEFINE_validator(metapath, &metapath_validator);
DEFINE_validator(step,     &integer_not_zero<uint32_t>);
DEFINE_validator(epoch,    &integer_not_zero<uint32_t>);

#define INFO_0(format, ...) do { \
    if (0 == plato::cluster_info_t::get_instance().partition_id_) { \
      int size = snprintf(nullptr, 0, format, ##__VA_ARGS__);       \
      std::string res;                                              \
      res.resize(size);                                             \
      snprintf(&res[0], size + 1, format, ##__VA_ARGS__);           \
      LOG(INFO) << res;                                             \
    }                                                               \
  } while(0)

/**
 * @brief
 * @param argc
 * @param argv
 */
void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.initialize(&argc, &argv);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "input:       " << FLAGS_input;
    LOG(INFO) << "types:       " << FLAGS_types;
    LOG(INFO) << "metapath:    " << FLAGS_metapath;
    LOG(INFO) << "output:      " << FLAGS_output;
    LOG(INFO) << "is_directed: " << FLAGS_is_directed;
    LOG(INFO) << "is_weighted: " << FLAGS_is_weighted;
    LOG(INFO) << "epoch:       " << FLAGS_epoch;
    LOG(INFO) << "step:        " << FLAGS_step;
    LOG(INFO) << "rate:        " << FLAGS_rate;
  }
}

const uint8_t META_CMD_MOVE            = 0x01;
const uint8_t META_CMD_KEEPFOOTPRINT   = 0x02;
const uint8_t META_CMD_FORCE_FLUSHPATH = 0x03;

using partition_t = plato::hash_by_source_t<>;
struct meta_walk_t {
  uint8_t command_;
};

/**
 * @brief
 * @tparam ENGINE
 * @tparam TYPE
 * @param engine
 * @param v_types
 */
template <typename ENGINE, typename TYPE>
void walk(ENGINE& engine, TYPE& v_types) {
  using walker_spec_t        = plato::walker_t<meta_walk_t>;
  using walk_context_spec_t  = plato::walk_context_t<meta_walk_t, typename ENGINE::partition_t>;

  auto& cluster_info = plato::cluster_info_t::get_instance();

  std::unique_ptr<plato::fs_mt_omp_output_t> output;
  if (0 != FLAGS_output.length()) {
    output.reset(new plato::fs_mt_omp_output_t(FLAGS_output,
        (boost::format("%04d_") % cluster_info.partition_id_).str(), true));
  }

  plato::walk_opts_t opts;
  opts.max_steps_ = FLAGS_step;
  opts.epochs_    = FLAGS_epoch;
  opts.start_rate = FLAGS_rate;

  auto* g_sampler = engine.sampler();
  
  print_mem_info("Before walk");
  engine.template walk<meta_walk_t>(
    [](walker_spec_t* pwalker) { },
    [&](walk_context_spec_t&& context, walker_spec_t& walker) {
      auto& wdata = walker.udata_;

      auto try_move = [&](void) {
        uint32_t next_type = _metapath[walker.step_id_ % _metapath.size()];
        auto choose_edge = g_sampler->sample(walker.current_v_id_, next_type, context.urng());
        if(choose_edge) {
          walker.current_v_id_ = choose_edge->neighbour_;
          return true;
        } else {
          return false;
        }
      };

      auto save_path = [&](void) {
        plato::footprint_t footprint = context.erase_footprint(walker);
        if (output) {
          auto& ostream = output->ostream();
          if (0 != footprint.idx_) {
            ostream << footprint.path_[0];
            for (plato::vid_t i = 1; i < footprint.idx_; ++i) {
              ostream << " " << footprint.path_[i];
            }
            ostream << "\n";
          }
        }
      };

      if (0 == walker.step_id_) {
        if (_metapath[0] == v_types[walker.current_v_id_]) {
          context.keep_footprint(walker);
          ++walker.step_id_;

          if (walker.step_id_ < opts.max_steps_ && try_move()) {
            context.keep_footprint(walker);
            ++walker.step_id_;

            if (walker.step_id_ < opts.max_steps_) {
              wdata.command_ = META_CMD_MOVE;
              context.move_to(walker.current_v_id_, walker);
            } else {
              save_path();
            }
          } else {
            save_path();
          }
        }
      } else {
        switch (wdata.command_) {
        case META_CMD_MOVE:
        {
          if (try_move()) {
            wdata.command_ = META_CMD_KEEPFOOTPRINT;
            context.move_to(walker.walk_id_, walker);

            ++walker.step_id_;
            if (walker.step_id_ < opts.max_steps_) {
              wdata.command_ = META_CMD_MOVE;
              context.move_to(walker.current_v_id_, walker);
            }
          } else {
            wdata.command_ = META_CMD_FORCE_FLUSHPATH;
            context.move_to(walker.walk_id_, walker);
          }

          break;
        }
        case META_CMD_KEEPFOOTPRINT:
        {
          context.keep_footprint(walker);
          if (walker.step_id_ + 1 >= opts.max_steps_) {
            save_path();
          }
          break;
        }
        case META_CMD_FORCE_FLUSHPATH:
          save_path();
          break;
        default:
          CHECK(false) << "unknown command: " << wdata.command_;
          break;
        }
      }
    }, opts);
}

/**
 * @brief
 */
void unbiased_walk(void) {
  //using walk_engine_spec_t = plato::walk_engine_t<plato::dummy_decoder<plato::empty_t>>;
  using part_spec_t         = plato::hash_by_source_t<>;
  using walk_engine_spec_t  = plato::walk_engine_t<plato::hnbbcsr_t<plato::empty_t, part_spec_t>, plato::empty_t>;

  plato::stop_watch_t watch;
  plato::graph_info_t graph_info(FLAGS_is_directed);

  watch.mark("t1");
  auto cache = plato::load_edges_cache<plato::empty_t, plato::vid_t, plato::edge_cache_t>(&graph_info, FLAGS_input,
    plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  auto& cluster_info = plato::cluster_info_t::get_instance();
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << graph_info.edges_;
    LOG(INFO) << "vertices:     " << graph_info.vertices_;
    LOG(INFO) << "max_v_id:     " << graph_info.max_v_i_;
    LOG(INFO) << "is_directed_: " << graph_info.is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }

  std::shared_ptr<partition_t> partitioner(new partition_t());
  plato::sparse_state_t<uint32_t, partition_t> v_types(graph_info.vertices_, partitioner);
  plato::load_vertices_state_from_path<uint32_t>(
    FLAGS_types, plato::edge_format_t::CSV,
    partitioner, plato::uint32_t_decoder,
    [&](plato::vertex_unit_t<uint32_t>&& unit) {
      v_types.insert(unit.vid_, unit.vdata_);
    });
  v_types.lock();
  walk_engine_spec_t engine(graph_info, *cache, v_types, partitioner);
  cache.reset();
  walk(engine, v_types);
  LOG(INFO) << "-----------------------------end------------------------------";
}

/**
 * @brief
 */
void biased_walk(void) {
  //using walk_engine_spec_t = plato::walk_engine_t<float>;
  using part_spec_t         = plato::hash_by_source_t<>;
  using walk_engine_spec_t  = plato::walk_engine_t<plato::hnbbcsr_t<float, part_spec_t>, float>;

  plato::stop_watch_t watch;
  plato::graph_info_t graph_info(FLAGS_is_directed);

  watch.mark("t1");

  std::shared_ptr<partition_t> partitioner(new partition_t());

  //print_mem_info("Before cache edges");
  auto cache = plato::load_edges_cache<float, plato::vid_t, plato::edge_cache_t>(&graph_info, FLAGS_input,
      plato::edge_format_t::CSV, plato::float_decoder);

  //print_mem_info("check edges");
  auto& cluster_info = plato::cluster_info_t::get_instance();
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << graph_info.edges_;
    LOG(INFO) << "vertices:     " << graph_info.vertices_;
    LOG(INFO) << "max_v_id:     " << graph_info.max_v_i_;
    LOG(INFO) << "is_directed_: " << graph_info.is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }

  plato::sparse_state_t<uint32_t, partition_t> v_types(graph_info.vertices_, partitioner);
  plato::load_vertices_state_from_path<uint32_t>(
    FLAGS_types, plato::edge_format_t::CSV,
    partitioner, plato::uint32_t_decoder,
    [&](plato::vertex_unit_t<uint32_t>&& unit) {
      v_types.insert(unit.vid_, unit.vdata_);
    });
  v_types.lock();

  walk_engine_spec_t engine(graph_info, *cache, v_types, partitioner);
  cache.reset();
  //print_mem_info("After cache set.;  Before walk.");
  walk(engine, v_types);
}

int main(int argc, char** argv) {
  init(argc, argv);

  plato::stop_watch_t watch;
  watch.mark("t1");
  
  if (FLAGS_is_weighted) {
    biased_walk();
  } else {
    unbiased_walk();
  }
  LOG(INFO) << "-----------------------------end------------------------------";
  INFO_0("total cost: %lfs", watch.show("t1") / 1000.0);
  return 0;
}
