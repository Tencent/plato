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

#include "plato/graph/state/vertex_cache.hpp"

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

std::vector<plato::vertex_unit_t<uint64_t>> g_vertices({
    plato::vertex_unit_t<uint64_t>{ 0, 8 },
    plato::vertex_unit_t<uint64_t>{ 1, 7 },
    plato::vertex_unit_t<uint64_t>{ 3, 6 },
    plato::vertex_unit_t<uint64_t>{ 3, 5 },
    plato::vertex_unit_t<uint64_t>{ 3, 4 },
    plato::vertex_unit_t<uint64_t>{ 6, 3 },
    plato::vertex_unit_t<uint64_t>{ 6, 2 },
    plato::vertex_unit_t<uint64_t>{ 8, 0 }
});

std::vector<plato::vertex_unit_t<non_trivial_t>> g_nontrivial_vertices({
    plato::vertex_unit_t<non_trivial_t>{ 0, non_trivial_t(0, 0.0) },
    plato::vertex_unit_t<non_trivial_t>{ 1, non_trivial_t(1, 1.0) },
    plato::vertex_unit_t<non_trivial_t>{ 3, non_trivial_t(2, 2.0) },
    plato::vertex_unit_t<non_trivial_t>{ 3, non_trivial_t(3, 3.0) },
    plato::vertex_unit_t<non_trivial_t>{ 3, non_trivial_t(4, 4.0) },
    plato::vertex_unit_t<non_trivial_t>{ 6, non_trivial_t(5, 5.0) },
    plato::vertex_unit_t<non_trivial_t>{ 6, non_trivial_t(6, 6.0) },
    plato::vertex_unit_t<non_trivial_t>{ 8, non_trivial_t(7, 7.0) },
    plato::vertex_unit_t<non_trivial_t>{ 8, non_trivial_t(8, 8.0) }
});

namespace plato {

template <typename T>
inline bool operator==(const plato::vertex_unit_t<T>& lhs, const plato::vertex_unit_t<T>& rhs) {
  return (lhs.vid_ == rhs.vid_) && (lhs.vdata_ == rhs.vdata_);
}

}

TEST(VertexCache, PushBackChunks) {
  ASSERT_TRUE(plato::vertex_cache_t<uint64_t>::is_trivial());
  plato::vertex_cache_t<uint64_t> cache(1024);

  cache.push_back(g_vertices.data(), g_vertices.size());
  ASSERT_EQ(cache.size(), g_vertices.size());

  for (size_t i = 0; i < g_vertices.size(); ++i) {
    ASSERT_EQ(cache[i], g_vertices[i]);
  }
}

TEST(VertexCache, PushBackChunksParallel) {
  ASSERT_FALSE(plato::vertex_cache_t<non_trivial_t>::is_trivial());
  plato::vertex_cache_t<non_trivial_t> cache(1024);

  std::vector<std::pair<size_t, size_t>> span({
    std::make_pair(0UL, 3UL), std::make_pair(3UL, 3UL), std::make_pair(6UL, 3UL)
  });

  #pragma omp parallel for num_threads(3)
  for (size_t i = 0; i < span.size(); ++i) {
    cache.push_back(&g_nontrivial_vertices[span[i].first], span[i].second);
  }
  ASSERT_EQ(cache.size(), g_nontrivial_vertices.size());

  for (size_t i = 0; i < g_nontrivial_vertices.size(); ++i) {
    ASSERT_THAT(g_nontrivial_vertices, testing::Contains(cache[i]));
  }
}

TEST(VertexCache, Traverse) {
  plato::vertex_cache_t<uint64_t> cache(1024);

  cache.push_back(g_vertices.data(), g_vertices.size());
  ASSERT_EQ(cache.size(), g_vertices.size());

  plato::vertex_cache_t<uint64_t> cache2(1024);

  cache.reset_traversal();
  #pragma omp parallel num_threads(3)
  {
    size_t chunk_size = 1;
    while (cache.next_chunk([&](size_t, plato::vertex_unit_t<uint64_t>* input) {
      cache2.push_back(*input);
    }, &chunk_size)) { }
  }

  ASSERT_EQ(cache2.size(), g_vertices.size());
  for (size_t i = 0; i < g_vertices.size(); ++i) {
    ASSERT_THAT(g_vertices, testing::Contains(cache2[i]));
  }
}

