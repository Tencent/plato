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

#include <cstdlib>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>

#include "mpi.h"
#include "omp.h"
#include "gtest/gtest.h"
#include "gtest_mpi_listener.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"

#include "plato/engine/dualmode.hpp"
#include "nstepdegrees.hpp"

using GRAPH_T     = std::pair<plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>,plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>>;
using bcsr_spec_t = plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>;
using dcsc_spec_t = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;
using v_subset_t  = plato::bitmap_t<>;

const std::string input = "data/graph/raw_graph_7_7.csv";     //data path.
const int32_t alpha = -1;                                     //alpha value used in sequence balance partition.
const bool is_directed = true;                                //is graph directed or not.
const bool part_by_in = true;                                 //partition by in-degree.
const int32_t step = 1;                                       //how many step's degree should be counted, -1 means infinity.
const int32_t bits = 6;                                       //hyperloglog bit width used for cardinality estimation.
const std::string type = "both";                                   //count 'in' degree or 'out' degree or 'both'.
const std::string actives = "ALL";          /*active vertex input in csv format, each line has one
                                                                vertex id. if this parameter is given, nstepdegrees
                                                                only calculate active vertex's nstepdegrees.*/

void init_cluster_info() {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  cluster_info.sockets_      = 1;
  cluster_info.threads_ = (int)std::strtol(std::getenv("OMP_NUM_THREADS"), nullptr, 10);
  omp_set_dynamic(0);
  omp_set_num_threads(cluster_info.threads_);
  cluster_info.sockets_ = numa_num_configured_nodes();
  MPI_Comm_size(MPI_COMM_WORLD, &cluster_info.partitions_);
  MPI_Comm_rank(MPI_COMM_WORLD, &cluster_info.partition_id_);
}
TEST(nstepdegrees, load_vertices) {
  init_cluster_info();

  plato::graph_info_t graph_info(is_directed);
  auto graph = plato::create_dualmode_seq_from_path<plato::empty_t>(&graph_info, input,
    plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>,
    alpha, part_by_in);

  std::vector<uint32_t> res;
  plato::load_vertices_state_from_path<uint32_t>("data/graph/raw_graph_2.csv",
    plato::edge_format_t::CSV, graph.second.partitioner(), plato::uint32_t_decoder,
    [&](plato::vertex_unit_t<uint32_t>&& unit) {
        res.push_back(unit.vid_);
    }
  );

  for(uint32_t i = 0; i < res.size(); ++i) {
      ASSERT_GE(6, res[i]);
  }
}

template <uint32_t BitWidth>
void test_work_flow(plato::dualmode_engine_t<dcsc_spec_t, bcsr_spec_t>* engine, const plato::graph_info_t& graph_info, GRAPH_T& graph, const v_subset_t& actives_v, const plato::algo::nstepdegree_opts_t& opts)
{
  plato::algo::nstepdegrees_t<dcsc_spec_t, bcsr_spec_t, BitWidth> nstepdegrees(engine, graph_info, actives_v, opts);
  nstepdegrees.compute(graph.second, graph.first);
  nstepdegrees.view_degrees();
  
  ASSERT_EQ(nstepdegrees.get_degree(0, true), 3); 
  ASSERT_EQ(nstepdegrees.get_degree(0, false), 1); 
  ASSERT_EQ(nstepdegrees.get_degree(4, true), 1); 
  ASSERT_EQ(nstepdegrees.get_degree(4, false), 1); 
}

TEST(nstepdegrees, computer) {
  init_cluster_info();

  plato::graph_info_t graph_info(is_directed);
  auto graph = plato::create_dualmode_seq_from_path<plato::empty_t>(&graph_info, input,
    plato::edge_format_t::CSV, plato::dummy_decoder<plato::empty_t>,
    alpha, part_by_in);

  plato::dualmode_engine_t<dcsc_spec_t, bcsr_spec_t> engine (
    std::shared_ptr<dcsc_spec_t>(&graph.second,  [](dcsc_spec_t*) { }),
    std::shared_ptr<bcsr_spec_t>(&graph.first, [](bcsr_spec_t*) { }),
    graph_info);
  
  auto actives_v = engine.alloc_v_subset();
  /*
  std::vector<uint32_t> res;
  plato::load_vertices_state_from_path<uint32_t>(actives, 
    plato::edge_format_t::CSV, graph.second.partitioner(), plato::uint32_t_decoder, 
    [&](plato::vertex_unit_t<uint32_t>&& unit) {
        res.push_back(unit.vid_);
    }
  );

  for(int i = 0; i < res.size(); ++i)
  {
    actives_v.set_bit(res[i]);
  }
  */
  if(actives != "ALL") {
    plato::load_vertices_state_from_path<uint32_t>(actives, 
    plato::edge_format_t::CSV, graph.second.partitioner(), plato::uint32_t_decoder, 
    [&](plato::vertex_unit_t<uint32_t>&& unit) {
      actives_v.set_bit(unit.vid_);
    });
  }
  else {
    actives_v.fill();
  }

  plato::algo::nstepdegree_opts_t opts;
  opts.step = step;
  opts.type = type;
  opts.is_directed = is_directed;
  
  switch (bits) {
    case 6:
      test_work_flow<6>(&engine, graph_info, graph, actives_v, opts);
      break;
    
    case 7:
      test_work_flow<7>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 8:
      test_work_flow<8>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 9:
      test_work_flow<9>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 10:
      test_work_flow<10>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 11:
      test_work_flow<11>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 12:
      test_work_flow<12>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 13:
      test_work_flow<13>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 14:
      test_work_flow<14>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 15:
      test_work_flow<15>(&engine, graph_info, graph, actives_v, opts);
      break;
    case 16:
      test_work_flow<16>(&engine, graph_info, graph, actives_v, opts);
      break;
      
    default:
      CHECK(false) << "unsupport hyperloglog bit width: " << bits
        << ", supported range is in [6, 16]";
  }
}

int main(int argc, char** argv)
{
    // Filter out Google Test arguments
  ::testing::InitGoogleTest(&argc, argv);

  // Initialize MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  // set OpenMP if not set
  if (getenv("OMP_NUM_THREADS") == nullptr) {
    setenv("OMP_NUM_THREADS", "2", 1);
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
    return RUN_ALL_TESTS();
}
