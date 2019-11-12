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

#include "plato/engine/walk.hpp"

#include <cstdlib>

#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gtest_mpi_listener.hpp"
#include "plato/graph/structure/hnbbcsr.hpp"
#include "plato/graph/graph.hpp"
#include "plato/util/spinlock.hpp"

void init_cluster_info(void) {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  cluster_info.partitions_   = 1;
  cluster_info.partition_id_ = 0;
  cluster_info.threads_      = 3;
  cluster_info.sockets_      = 1;
}

TEST(Walk, InitFromUnweightedUndirected) {
  init_cluster_info();

  using part_spec_t         = plato::hash_by_source_t<>;
  using walk_engine_spec_t  = plato::walk_engine_t<plato::cbcsr_t<plato::empty_t, part_spec_t>, plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = false;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/non_coding_5_7.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  ASSERT_NO_THROW({
    walk_engine_spec_t engine(graph_info, *cache, partitioner);
  });
}

TEST(Walk, InitFromUnweightedDirected) {
  init_cluster_info();

  using part_spec_t         = plato::hash_by_source_t<>;
  using walk_engine_spec_t  = plato::walk_engine_t<plato::cbcsr_t<plato::empty_t, part_spec_t>, plato::empty_t>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/non_coding_5_8.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);

  ASSERT_NO_THROW({
    walk_engine_spec_t engine(graph_info, *cache, partitioner);
  });
}

TEST(Walk, InitFromWeightedUndirected) {
  init_cluster_info();

  using part_spec_t         = plato::hash_by_source_t<>;
  using walk_engine_spec_t  = plato::walk_engine_t<plato::cbcsr_t<float, part_spec_t>, float>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = false;

  auto cache = plato::load_edges_cache<float>(&graph_info, "data/graph/non_coding_5_7_weighted.csv",
      plato::edge_format_t::CSV, plato::float_decoder);

  ASSERT_NO_THROW({
    walk_engine_spec_t engine(graph_info, *cache, partitioner);
  });
}

TEST(Walk, InitFromWeightedDirected) {
  init_cluster_info();

  using part_spec_t         = plato::hash_by_source_t<>;
  using walk_engine_spec_t  = plato::walk_engine_t<plato::cbcsr_t<float, part_spec_t>, float>;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<float>(&graph_info, "data/graph/non_coding_5_8_weighted.csv",
      plato::edge_format_t::CSV, plato::float_decoder);

  ASSERT_NO_THROW({
    walk_engine_spec_t engine(graph_info, *cache, partitioner);
  });
}

TEST(Walk, RandomWalk) {
  init_cluster_info();

  using part_spec_t         = plato::hash_by_source_t<>;
  using walk_engine_spec_t  = plato::walk_engine_t<plato::cbcsr_t<float, part_spec_t>, float>;

  using walker_spec_t       = plato::walker_t<uint8_t>;
  using walk_context_spec_t = plato::walk_context_t<uint8_t, part_spec_t>;

  plato::spinlock_noaligned_t show_lock;

  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;

  auto cache = plato::load_edges_cache<float>(&graph_info, "data/graph/non_coding_5_8_weighted.csv",
      plato::edge_format_t::CSV, plato::float_decoder);

  walk_engine_spec_t engine(graph_info, *cache, partitioner);

  plato::walk_opts_t opts;
  opts.max_steps_ = 3;
  opts.epochs_    = 3;
  opts.start_rate = 0.02;

  auto* g_sampler = engine.sampler();

  engine.walk<uint8_t>(
    [&](walker_spec_t* walker) { walker->udata_ = 0; },
    [&](walk_context_spec_t&& context, walker_spec_t& walker) {
      auto made_proposal = [&](void) {
        auto choose_edge = g_sampler->sample(walker.current_v_id_, context.urng());
        CHECK(choose_edge != NULL);
        plato::vid_t proposal = choose_edge->neighbour_;
        
        ++walker.step_id_;
        walker.current_v_id_ = proposal;
      };

      if (0 == walker.step_id_) {
        context.keep_footprint(walker);
        made_proposal();

        // keep footprint
        context.keep_footprint(walker);

        // move to next
        walker.udata_ = 0x02;
        context.move_to(walker.current_v_id_, walker);
      } else if (1 == walker.step_id_) {
        switch (walker.udata_) {
        case 0x01:
          context.keep_footprint(walker);
          break;
        case 0x02:
        {
          made_proposal();

          // save
          walker.udata_ = 0x03;
          context.move_to(walker.walk_id_, walker);

          // move
          walker.udata_ = 0x02;
          context.move_to(walker.current_v_id_, walker);
          break;
        }
        default:
          CHECK(false) << "unknown command: " << walker.udata_;
          break;
        }
      } else {
        if (0x03 == walker.udata_) {
          context.keep_footprint(walker);
          auto footprint = context.erase_footprint(walker);

          (void)footprint;
          show_lock.lock();
          LOG(INFO) << "-----------------------RandomWalk Start--------------------------";
          LOG(INFO) << "walker " << walker.walk_id_ << " : ";
          for(plato::vid_t i = 0; i < footprint.idx_; ++i) {
            LOG(INFO) << (footprint.path_)[i];
          }
          LOG(INFO) << "walker " << walker.walk_id_ << " end----";
          LOG(INFO) << "-----------------------RandomWalk End----------------------------";
          show_lock.unlock();
        }
      }
    },
    opts);
}

TEST(Walk, RandomWalkOnMultiTypes) {
  init_cluster_info();

  using part_spec_t         = plato::hash_by_source_t<>;
  using walk_engine_spec_t  = plato::walk_engine_t<plato::hnbbcsr_t<plato::empty_t, part_spec_t>, plato::empty_t>;
  using walker_spec_t       = plato::walker_t<uint8_t>;
  using walk_context_spec_t = plato::walk_context_t<uint8_t, part_spec_t>;

  plato::spinlock_noaligned_t show_lock;

  plato::graph_info_t graph_info;
  graph_info.is_directed_ = true;
  std::shared_ptr<part_spec_t> partitioner(new part_spec_t());
  auto cache = plato::load_edges_cache<plato::empty_t>(&graph_info, "data/graph/non_coding_5_8.csv",
      plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>);
  plato::sparse_state_t<uint32_t, part_spec_t> v_types(graph_info.vertices_, partitioner);

  plato::load_vertices_state_from_path<uint32_t>("data/graph/non_coding_5_8_vertic.csv", plato::edge_format_t::CSV,
      partitioner, plato::uint32_t_decoder,
      [&](plato::vertex_unit_t<uint32_t>&& unit) {
        v_types.insert(unit.vid_, unit.vdata_);
      });
  v_types.lock();

  walk_engine_spec_t engine(graph_info, *cache, v_types, partitioner);

  plato::walk_opts_t opts;
  opts.max_steps_ = 3;
  opts.epochs_    = 3;
  opts.start_rate = 0.02;

  auto* g_sampler = engine.sampler();
  engine.walk<uint8_t>(
    [&](walker_spec_t* walker) { walker->udata_ = 0; },
    [&](walk_context_spec_t&& context, walker_spec_t& walker) {
      auto made_proposal = [&](void) {
        uint32_t next_type = rand() % 4;
        auto choose_edge = g_sampler->sample(walker.current_v_id_, next_type, context.urng());
        if(choose_edge) {
          walker.current_v_id_ = choose_edge->neighbour_;
        } 

        ++walker.step_id_;
      };
      if (0 == walker.step_id_) {
        context.keep_footprint(walker);
        made_proposal();
        // keep footprint
        context.keep_footprint(walker);
        // move to next
        walker.udata_ = 0x02;
        context.move_to(walker.current_v_id_, walker);
      } else if (1 == walker.step_id_) {
        switch (walker.udata_) {
        case 0x01:
          context.keep_footprint(walker);
          break;
        case 0x02:
        {
          made_proposal();
          // save
          walker.udata_ = 0x03;
          context.move_to(walker.walk_id_, walker);

          // move
          walker.udata_ = 0x02;
          context.move_to(walker.current_v_id_, walker);
          break;
        }
        default:
          CHECK(false) << "unknown command: " << walker.udata_;
          break;
        }
      } else {
        if (0x03 == walker.udata_) {
          context.keep_footprint(walker);
          auto footprint = context.erase_footprint(walker);

          (void)footprint;
          show_lock.lock();
          LOG(INFO) << "-----------------------RandomWalkOnMultiTypes Start--------------------------";
          LOG(INFO) << "walker " << walker.walk_id_ << " : ";
          for(plato::vid_t i = 0; i < footprint.idx_; ++i) {
            LOG(INFO) << (footprint.path_)[i];
          }
          LOG(INFO) << "walker " << walker.walk_id_ << " end----";
          LOG(INFO) << "-----------------------RandomWalkOnMultiTypes End----------------------------";
          show_lock.unlock();
        }
      }
    },
    opts);
}

int main(int argc, char** argv) {
  // Filter out Google Test arguments
  ::testing::InitGoogleTest(&argc, argv);

  google::InitGoogleLogging("plato-test");
  google::LogToStderr();

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  // set OpenMP if not set
  if (nullptr == getenv("OMP_NUM_THREADS")) {
    setenv("OMP_NUM_THREADS", "3", 1);
  }

  // Add object that will finalize MPI on exit; Google Test owns this pointer
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

  // Get the event listener list.
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();

  // Remove default listener
  delete listeners.Release(listeners.default_result_printer());

  // Adds MPI listener; Google Test owns this pointer
  listeners.Append(new MPIMinimalistPrinter);

  // Run tests, then clean up and exit
  return RUN_ALL_TESTS();
}

