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

#include "plato/graph/detail/structure/edge/adjlist.hpp"

#include <vector>
#include <utility>

#include "gtest/gtest.h"

using namespace plato;

std::vector<edge_unit_t<empty_t>> g_weightless_edges({
  edge_unit_t<empty_t>{ 0, 8 },
  edge_unit_t<empty_t>{ 0, 7 },
  edge_unit_t<empty_t>{ 3, 6 },
  edge_unit_t<empty_t>{ 4, 5 },
  edge_unit_t<empty_t>{ 5, 4 },
  edge_unit_t<empty_t>{ 6, 3 },
  edge_unit_t<empty_t>{ 7, 2 },
  edge_unit_t<empty_t>{ 8, 0 }
});

std::vector<edge_unit_t<uint64_t>> g_edges({
  edge_unit_t<uint64_t>{ 0, 8, 0 },
  edge_unit_t<uint64_t>{ 1, 7, 1 },
  edge_unit_t<uint64_t>{ 3, 6, 2 },
  edge_unit_t<uint64_t>{ 3, 5, 3 },
  edge_unit_t<uint64_t>{ 3, 4, 4 },
  edge_unit_t<uint64_t>{ 6, 3, 5 },
  edge_unit_t<uint64_t>{ 6, 2, 6 },
  edge_unit_t<uint64_t>{ 8, 0, 7 }
});

void init_graph_info(graph_info_t* pginfo) {
  cluster_info_t& pcinfo = cluster_info_t::get_instance();

  pcinfo.partitions_   = 1;
  pcinfo.partition_id_ = 0;
  pcinfo.threads_      = 1;
  pcinfo.sockets_      = 1;

  pginfo->is_directed_ = true;
}

void check_weightless_edges(adjlist_t<empty_t>& adjlist) {
  adj_unit_list_t<empty_t> adjs;
  ASSERT_EQ(0, adjlist.from(&adjs, 0));
  ASSERT_EQ(2, adjs.end_ - adjs.begin_);
  ASSERT_EQ(8, adjs.begin_[0].neighbour_);
  ASSERT_EQ(7, adjs.begin_[1].neighbour_);

  ASSERT_EQ(0, adjlist.from(&adjs, 3));
  ASSERT_EQ(1, adjs.end_ - adjs.begin_);
  ASSERT_EQ(6, adjs.begin_[0].neighbour_);

  ASSERT_EQ(0, adjlist.from(&adjs, 4));
  ASSERT_EQ(1, adjs.end_ - adjs.begin_);
  ASSERT_EQ(5, adjs.begin_[0].neighbour_);

  ASSERT_EQ(0, adjlist.from(&adjs, 5));
  ASSERT_EQ(1, adjs.end_ - adjs.begin_);
  ASSERT_EQ(4, adjs.begin_[0].neighbour_);

  ASSERT_EQ(0, adjlist.from(&adjs, 6));
  ASSERT_EQ(1, adjs.end_ - adjs.begin_);
  ASSERT_EQ(3, adjs.begin_[0].neighbour_);

  ASSERT_EQ(0, adjlist.from(&adjs, 7));
  ASSERT_EQ(1, adjs.end_ - adjs.begin_);
  ASSERT_EQ(2, adjs.begin_[0].neighbour_);

  ASSERT_EQ(0, adjlist.from(&adjs, 8));
  ASSERT_EQ(1, adjs.end_ - adjs.begin_);
  ASSERT_EQ(0, adjs.begin_[0].neighbour_);

  ASSERT_EQ(-1, adjlist.from(&adjs, 11));
}

void check_edges(adjlist_t<uint64_t>& adjlist) {
  adj_unit_list_t<uint64_t> adjs;

  ASSERT_EQ(0, adjlist.from(&adjs, 0));
  ASSERT_EQ(1, adjs.end_ - adjs.begin_);
  ASSERT_EQ(8, adjs.begin_[0].neighbour_);
  ASSERT_EQ(0, adjs.begin_[0].edata_);

  ASSERT_EQ(0, adjlist.from(&adjs, 1));
  ASSERT_EQ(1, adjs.end_ - adjs.begin_);
  ASSERT_EQ(7, adjs.begin_[0].neighbour_);
  ASSERT_EQ(1, adjs.begin_[0].edata_);

  ASSERT_EQ(0, adjlist.from(&adjs, 3));
  ASSERT_EQ(3, adjs.end_ - adjs.begin_);
  ASSERT_EQ(6, adjs.begin_[0].neighbour_);
  ASSERT_EQ(2, adjs.begin_[0].edata_);
  ASSERT_EQ(5, adjs.begin_[1].neighbour_);
  ASSERT_EQ(3, adjs.begin_[1].edata_);
  ASSERT_EQ(4, adjs.begin_[2].neighbour_);
  ASSERT_EQ(4, adjs.begin_[2].edata_);

  ASSERT_EQ(0, adjlist.from(&adjs, 6));
  ASSERT_EQ(2, adjs.end_ - adjs.begin_);
  ASSERT_EQ(3, adjs.begin_[0].neighbour_);
  ASSERT_EQ(5, adjs.begin_[0].edata_);
  ASSERT_EQ(2, adjs.begin_[1].neighbour_);
  ASSERT_EQ(6, adjs.begin_[1].edata_);

  ASSERT_EQ(0, adjlist.from(&adjs, 8));
  ASSERT_EQ(1, adjs.end_ - adjs.begin_);
  ASSERT_EQ(0, adjs.begin_[0].neighbour_);
  ASSERT_EQ(7, adjs.begin_[0].edata_);

  ASSERT_EQ(-1, adjlist.from(&adjs, 11));
}

TEST(EStoreAdjlist, Init) {
  ASSERT_NO_THROW({
    adjlist_t<empty_t> adjlist;
  });

  ASSERT_NO_THROW({
    adjlist_t<double> adjlist;
  });
}

TEST(EStoreAdjlist, AddEdgeOnebyone) {
  adjlist_t<empty_t> adjlist;

  graph_info_t graph_info;
  init_graph_info(&graph_info);

  adjlist.initialize(graph_info);
  for (const auto& edge: g_weightless_edges) {
    ASSERT_EQ(0, adjlist.add_edge(edge));
  }
  adjlist.finalize(graph_info);

  // check edges
  check_weightless_edges(adjlist);
}

TEST(EStoreAdjlist, AddEdgeChunks) {
  adjlist_t<uint64_t> adjlist;

  graph_info_t graph_info;
  init_graph_info(&graph_info);

  adjlist.initialize(graph_info);
  ASSERT_EQ(0, adjlist.add_edges(g_edges));
  adjlist.finalize(graph_info);

  // check edges
  check_edges(adjlist);
}

TEST(EStoreAdjlist, Traverse) {
  using estore_t = adjlist_t<uint64_t>;

  estore_t adjlist;

  graph_info_t graph_info;
  init_graph_info(&graph_info);

  adjlist.initialize(graph_info);
  ASSERT_EQ(0, adjlist.add_edges(g_edges));
  adjlist.finalize(graph_info);

  auto traversal = adjlist.get_traversal();
  std::pair<vid_t, estore_t::adj_unit_list_spec_t> v_edges;
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(traversal.next(&v_edges));
    std::for_each(v_edges.second.begin_, v_edges.second.end_,
      [&](estore_t::adj_unit_spec_t& edge) {
        bool found = false;
        std::for_each(g_edges.begin(), g_edges.end(),
          [&](edge_unit_t<uint64_t>& raw_edge) {
            if (raw_edge.src_ == v_edges.first &&
                raw_edge.dst_ == edge.neighbour_ &&
                raw_edge.edata_ == edge.edata_) {
              found = true;
            }
          }
        );
        ASSERT_TRUE(found);
      }
    );
  }
  ASSERT_FALSE(traversal.next(&v_edges));
}

TEST(EStoreAdjlist, TraverseChunk) {
  using estore_t = adjlist_t<empty_t>;

  estore_t adjlist;

  graph_info_t graph_info;
  init_graph_info(&graph_info);
  cluster_info_t::get_instance().threads_ = 2;

  adjlist.initialize(graph_info);
  ASSERT_EQ(0, adjlist.add_edges(g_weightless_edges));
  adjlist.finalize(graph_info);

  int  e_count    = 0;
  auto traversals = adjlist.get_traversals();
  ASSERT_EQ(2, traversals.size());

  for (auto& traversal: traversals) {
    std::vector<std::pair<vid_t, estore_t::adj_unit_list_spec_t>> adjs;
    while (traversal.next_chunk(&adjs, 2)) {
      ASSERT_TRUE(adjs.size() <= 2);
      for (auto& v_edges: adjs) {
        std::for_each(v_edges.second.begin_, v_edges.second.end_,
          [&](estore_t::adj_unit_spec_t& edge) {
            ++e_count;
            bool found = false;
            std::for_each(g_weightless_edges.begin(), g_weightless_edges.end(),
              [&](estore_t::edge_unit_spec_t& raw_edge) {
                if (raw_edge.src_ == v_edges.first && raw_edge.dst_ == edge.neighbour_) {
                  found = true;
                }
              }
            );
            ASSERT_TRUE(found);
          }
        );
      }
    }
  }
  ASSERT_EQ(8, e_count);
}

