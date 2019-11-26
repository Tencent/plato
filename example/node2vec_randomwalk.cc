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

DEFINE_string(input,       "",     "input graph file, in csv format, eg: src,dst[,weight]");
DEFINE_string(output,      "",     "output path, in csv format, gzip compressed");
DEFINE_bool(is_weighted,   false,  "random walk with bias or not");
DEFINE_double(p,           1.0,    "backward bias for randomwalk");
DEFINE_double(q,           0.5,    "forward bias for randomwalk");
DEFINE_uint32(epoch,       1,      "how many epoch should perform");
DEFINE_uint32(step,        10,     "steps per epoch");
DEFINE_double(rate,        0.02,   "start 'rate'% walker per one bsp, reduce memory consumption");

bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

DEFINE_validator(input,  &string_not_empty);

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

using partition_t = plato::hash_by_source_t<>;

#define N2V_COMMAND_MOVE          (0x01)
#define N2V_COMMAND_IS_NEIGHBOR   (0x02)
#define N2V_COMMAND_RESPONSE      (0x03)
#define N2V_COMMAND_KEEPFOOTPRINT (0x04)

struct n2v_walk_t {
  bool           is_negb_;
  uint8_t        command_;
  plato::vid_t   proposal_;
  float          prob_;
  plato::vid_t   from_;
};

/*
 * Implement Node2vec with rejection sampling 
 *
 * reference:
 * Ke Yang, MingXing Zhang, Kang Chen, Xiaosong Ma, Yang Bai, and Yong Jiang. 2019.
 * KnightKing: A Fast Distributed Graph Ran- dom Walk Engine.
 * In ACM SIGOPS 27th Symposium on Operating Systems Principles (SOSP ’19),
 * October 27–30, 2019
 *
 * */
template <typename ENGINE>
void walk(ENGINE& engine) {
  using walker_spec_t        = plato::walker_t<n2v_walk_t>;
  using walk_context_spec_t  = plato::walk_context_t<n2v_walk_t, typename ENGINE::partition_t>;

  auto& cluster_info = plato::cluster_info_t::get_instance();
  float upbnd = std::max(1.0, std::max(1.0 / FLAGS_p, 1.0 / FLAGS_q));  // upper bound
  float lwbnd = std::min(1.0, std::min(1.0 / FLAGS_p, 1.0 / FLAGS_q));  // lower bound

  // init output
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

  engine.template walk<n2v_walk_t>(
    [&](walker_spec_t*) { },
    [&](walk_context_spec_t&& context, walker_spec_t& walker) {
      auto& wdata = walker.udata_;

      auto is_terminate = [&](void) {
        return walker.step_id_ >= opts.max_steps_;
      };

      auto output_footprint = [&](plato::footprint_t& footprint) {
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

      auto oot = [&](void) {  // output when terminated
        if (is_terminate()) {
          plato::footprint_t footprint = context.erase_footprint(walker);
          output_footprint(footprint);
          return true;
        }
        return false;
      };

      auto make_proposal = [&](void) {
        std::uniform_real_distribution<float> dist(0, upbnd);

        auto choose_edge = g_sampler->sample(walker.current_v_id_, context.urng());
        CHECK(choose_edge != NULL);
        plato::vid_t proposal = choose_edge->neighbour_;

        float prob = dist(context.urng());

        wdata.proposal_ = proposal;
        wdata.prob_     = prob;
      };

      auto accept_proposal = [&](void) {
        ++walker.step_id_;
        wdata.from_ = walker.current_v_id_;
        walker.current_v_id_ = wdata.proposal_;
      };

      auto akm = [&](void) {  // accept && keep_footprint && move
        accept_proposal();
        wdata.command_ = N2V_COMMAND_KEEPFOOTPRINT;
        context.move_to(walker.walk_id_, walker);
        if (false == is_terminate()) {
          wdata.command_ = N2V_COMMAND_MOVE;
          context.move_to(walker.current_v_id_, walker);
        }
      };

      if (0 == walker.step_id_) {
        if (is_terminate()) { return ; }  // nonsense
        ++walker.step_id_;
        context.keep_footprint(walker);
        if (oot()) { return ; }

        make_proposal();
        accept_proposal();
        context.keep_footprint(walker);
        if (oot()) { return ; }

        wdata.command_ = N2V_COMMAND_MOVE;
        context.move_to(walker.current_v_id_, walker);
      } else {
        switch (wdata.command_) {
        case N2V_COMMAND_MOVE:
        {
          while (true) {
            make_proposal();
            if (wdata.prob_ < lwbnd) {  // accept directly
              akm();
              break;
            } else if (wdata.proposal_ == wdata.from_) {  // proposal_ is last visted vertex
              if (wdata.prob_ < (1.0 / FLAGS_p)) {  // accept
                akm();
                break;
              }
            } else {  // ask last visted vertex
              wdata.command_ = N2V_COMMAND_IS_NEIGHBOR;
              context.move_to(wdata.from_, walker);
              break;
            }
          }

          break;
        }
        case N2V_COMMAND_IS_NEIGHBOR:
        {
          wdata.is_negb_ = g_sampler->existed(wdata.from_, wdata.proposal_);
          wdata.command_ = N2V_COMMAND_RESPONSE;
          context.move_to(walker.current_v_id_, walker);

          break;
        }
        case N2V_COMMAND_RESPONSE:
        {
          if (wdata.is_negb_) {
            if (wdata.prob_ < 1.0) {
              akm();
            } else {  // proposal again
              wdata.command_ = N2V_COMMAND_MOVE;
              context.move_to(walker.current_v_id_, walker);
            }
          } else {
            if (wdata.prob_ < (1.0 / FLAGS_q)) {
              akm();
            } else {  // proposal again
              wdata.command_ = N2V_COMMAND_MOVE;
              context.move_to(walker.current_v_id_, walker);
            }
          }

          break;
        }
        case N2V_COMMAND_KEEPFOOTPRINT:
        {
          context.keep_footprint(walker);
          oot();
          break;
        }
        default:
          CHECK(false) << "unknown command: " << wdata.command_;
          break;
        }
      }
    }, opts);
}

void biased_walk(void) {
  //using walk_engine_spec_t = plato::walk_engine_t<true>;
  using part_spec_t         = plato::hash_by_source_t<>;
  using walk_engine_spec_t  = plato::walk_engine_t<plato::cbcsr_t<float, part_spec_t>, float>;

  plato::stop_watch_t watch;
  plato::graph_info_t graph_info(false);

  watch.mark("t1");
  auto cache = plato::load_edges_cache<float, plato::vid_t, plato::edge_cache_t>(&graph_info, FLAGS_input, plato::edge_format_t::CSV,
      plato::float_decoder);

  auto& cluster_info = plato::cluster_info_t::get_instance();
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << graph_info.edges_;
    LOG(INFO) << "vertices:     " << graph_info.vertices_;
    LOG(INFO) << "max_v_id:     " << graph_info.max_v_i_;
    LOG(INFO) << "is_directed_: " << graph_info.is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }

  std::shared_ptr<partition_t> partitioner(new partition_t());
  //walk_engine_spec_t engine(graph_info, *cache, partitioner);
  walk_engine_spec_t engine(graph_info, *cache, partitioner);
  cache.reset();
  walk(engine);
}

void unbiased_walk(void) {
  //using walk_engine_spec_t = plato::walk_engine_t<false>;
  using part_spec_t         = plato::hash_by_source_t<>;
  using walk_engine_spec_t  = plato::walk_engine_t<plato::cbcsr_t<plato::empty_t, part_spec_t>, plato::empty_t>;

  plato::stop_watch_t watch;
  plato::graph_info_t graph_info(false);

  watch.mark("t1");
  auto cache = plato::load_edges_cache<plato::empty_t, plato::vid_t, plato::edge_cache_t>(&graph_info, FLAGS_input, plato::edge_format_t::CSV,
      plato::dummy_decoder<plato::empty_t>);

  auto& cluster_info = plato::cluster_info_t::get_instance();
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << graph_info.edges_;
    LOG(INFO) << "vertices:     " << graph_info.vertices_;
    LOG(INFO) << "max_v_id:     " << graph_info.max_v_i_;
    LOG(INFO) << "is_directed_: " << graph_info.is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }

  std::shared_ptr<partition_t> partitioner(new partition_t());
  walk_engine_spec_t engine(graph_info, *cache, partitioner);
  cache.reset();
  walk(engine);
}

int main(int argc, char** argv) {
  init(argc, argv);

  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();
  cluster_info.initialize(&argc, &argv);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "input:       " << FLAGS_input;
    LOG(INFO) << "output:      " << FLAGS_output;
    LOG(INFO) << "is_weighted: " << FLAGS_is_weighted;
    LOG(INFO) << "p:           " << FLAGS_p;
    LOG(INFO) << "q:           " << FLAGS_q;
    LOG(INFO) << "epoch:       " << FLAGS_epoch;
    LOG(INFO) << "step:        " << FLAGS_step;
    LOG(INFO) << "rate:        " << FLAGS_rate;
  }

  if (FLAGS_is_weighted) {
    biased_walk();
  } else {
    unbiased_walk();
  }

  return 0;
}

