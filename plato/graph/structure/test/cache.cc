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

#include "plato/graph/structure/cache.hpp"
#include "plato/graph/base.hpp"

#include <cstdint>
#include <vector>
#include <utility>
#include <algorithm>

#include "omp.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

struct non_trivial_t {
  non_trivial_t(void): a_(0), b_(0.0) { }
  non_trivial_t(uint64_t a, double b): a_(a), b_(b) { }

  bool operator==(const non_trivial_t& other) const {
    return (a_ == other.a_) && (b_ == other.b_);
  }

  uint64_t a_;
  double   b_;
};

std::vector<plato::edge_unit_t<plato::empty_t>> g_weightless_edges({
    plato::edge_unit_t<plato::empty_t>{ 0, 8 },
    plato::edge_unit_t<plato::empty_t>{ 0, 7 },
    plato::edge_unit_t<plato::empty_t>{ 3, 6 },
    plato::edge_unit_t<plato::empty_t>{ 4, 5 },
    plato::edge_unit_t<plato::empty_t>{ 5, 4 },
    plato::edge_unit_t<plato::empty_t>{ 6, 3 },
    plato::edge_unit_t<plato::empty_t>{ 7, 2 },
    plato::edge_unit_t<plato::empty_t>{ 8, 0 }
});

std::vector<plato::edge_unit_t<uint64_t>> g_edges({
    plato::edge_unit_t<uint64_t>{ 0, 8, 0 },
    plato::edge_unit_t<uint64_t>{ 1, 7, 1 },
    plato::edge_unit_t<uint64_t>{ 3, 6, 2 },
    plato::edge_unit_t<uint64_t>{ 3, 5, 3 },
    plato::edge_unit_t<uint64_t>{ 3, 4, 4 },
    plato::edge_unit_t<uint64_t>{ 6, 3, 5 },
    plato::edge_unit_t<uint64_t>{ 6, 2, 6 },
    plato::edge_unit_t<uint64_t>{ 8, 0, 7 }
});

std::vector<plato::edge_unit_t<non_trivial_t>> g_nontrivial_edges({
    plato::edge_unit_t<non_trivial_t>{ 0, 8, non_trivial_t(0, 0.0) },
    plato::edge_unit_t<non_trivial_t>{ 1, 7, non_trivial_t(1, 1.0) },
    plato::edge_unit_t<non_trivial_t>{ 3, 6, non_trivial_t(2, 2.0) },
    plato::edge_unit_t<non_trivial_t>{ 3, 5, non_trivial_t(3, 3.0) },
    plato::edge_unit_t<non_trivial_t>{ 3, 4, non_trivial_t(4, 4.0) },
    plato::edge_unit_t<non_trivial_t>{ 6, 3, non_trivial_t(5, 5.0) },
    plato::edge_unit_t<non_trivial_t>{ 6, 2, non_trivial_t(6, 6.0) },
    plato::edge_unit_t<non_trivial_t>{ 8, 0, non_trivial_t(7, 7.0) },
    plato::edge_unit_t<non_trivial_t>{ 8, 1, non_trivial_t(8, 8.0) }
});

namespace plato {

inline bool operator==(const plato::edge_unit_t<plato::empty_t>& lhs, const plato::edge_unit_t<plato::empty_t>& rhs) {
  return (lhs.src_ == rhs.src_) && (lhs.dst_ == rhs.dst_);
}

template <typename T>
inline bool operator==(const plato::edge_unit_t<T>& lhs, const plato::edge_unit_t<T>& rhs) {
  return (lhs.src_ == rhs.src_) && (lhs.dst_ == rhs.dst_) && (lhs.edata_ == rhs.edata_);
}

}

TEST(EdgeCache, PushBackOneByOne) {
  ASSERT_TRUE(plato::cache_t<plato::edge_unit_t<plato::empty_t>>::is_trivial());

  plato::cache_t<plato::edge_unit_t<plato::empty_t>> cache(1024);

  for (const auto& edge: g_weightless_edges) {
    cache.push_back(edge);
  }
  ASSERT_EQ(cache.size_, g_weightless_edges.size());

  for (size_t i = 0; i < g_weightless_edges.size(); ++i) {
    ASSERT_EQ(cache.data_.get()[i], g_weightless_edges[i]);
  }
}

TEST(EdgeCache, PushBackOneByOneParallel) {
  plato::cache_t<plato::edge_unit_t<plato::empty_t>> cache(1024);

  #pragma omp parallel for num_threads(3)
  for (size_t i = 0; i < g_weightless_edges.size(); ++i) {
    cache.push_back(g_weightless_edges[i]);
  }
  ASSERT_EQ(cache.size_, g_weightless_edges.size());

  for (size_t i = 0; i < g_weightless_edges.size(); ++i) {
    ASSERT_THAT(g_weightless_edges, testing::Contains(cache.data_.get()[i]));
  }
}

TEST(EdgeCache, PushBackChunks) {
  ASSERT_TRUE(plato::cache_t<plato::edge_unit_t<uint64_t>>::is_trivial());
  plato::cache_t<plato::edge_unit_t<uint64_t>> cache(1024);

  cache.push_back(g_edges.data(), g_edges.size());
  ASSERT_EQ(cache.size_, g_edges.size());

  for (size_t i = 0; i < g_edges.size(); ++i) {
    ASSERT_EQ(cache.data_.get()[i], g_edges[i]);
  }
}

TEST(EdgeCache, PushBackChunksParallel) {
  ASSERT_FALSE(plato::cache_t<plato::edge_unit_t<non_trivial_t>>::is_trivial());
  plato::cache_t<plato::edge_unit_t<non_trivial_t>> cache(1024);

  std::vector<std::pair<size_t, size_t>> span({
    std::make_pair(0UL, 3UL), std::make_pair(3UL, 3UL), std::make_pair(6UL, 3UL)
  });

  #pragma omp parallel for num_threads(3)
  for (size_t i = 0; i < span.size(); ++i) {
    cache.push_back(&g_nontrivial_edges[span[i].first], span[i].second);
  }
  ASSERT_EQ(cache.size_, g_nontrivial_edges.size());

  for (size_t i = 0; i < g_nontrivial_edges.size(); ++i) {
    ASSERT_THAT(g_nontrivial_edges, testing::Contains(cache.data_.get()[i]));
  }
}

TEST(EdgeCache, ForeachEdges) {
  plato::cache_t<plato::edge_unit_t<uint64_t>> cache(1024);

  cache.push_back(g_edges.data(), g_edges.size());
  ASSERT_EQ(cache.size_, g_edges.size());

  plato::cache_t<plato::edge_unit_t<uint64_t>> cache2(1024);
  cache.foreach([&cache2](plato::edge_unit_t<uint64_t>* input, size_t length) {
    for (size_t i = 0; i < length; ++i) {
      cache2.push_back(input[i]);
    }
    return true;
  }, 3);
  ASSERT_EQ(cache2.size_, g_edges.size());

  for (size_t i = 0; i < g_edges.size(); ++i) {
    ASSERT_THAT(g_edges, testing::Contains(cache2.data_.get()[i]));
  }
}

