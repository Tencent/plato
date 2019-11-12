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

#include "plato/util/mmap_alloc.hpp"

#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

struct non_trivial_t {
  non_trivial_t(int a): a_(0) {  }

  bool operator==(const non_trivial_t& other) { return a_ == other.a_; }
  bool operator==(const int& a) { return a_ == a; }

  int a_;
};

TEST(MMAP_ALLOCATOR, Init) {
  plato::mmap_allocator_t<int> alloc();
}

TEST(MMAP_ALLOCATOR, VectorWithTrivialType) {
  std::vector<int, plato::mmap_allocator_t<int>> vec;

  for (int i = 0; i < 1033; ++i) {
    vec.emplace_back(i);
  }
  ASSERT_EQ(vec.size(), 1033);

  for (size_t i = 0; i < vec.size(); ++i) {
    ASSERT_THAT(vec, testing::Contains((int)i));
  }
}

TEST(MMAP_ALLOCATOR, VectorWithNonTrivialType) {
  std::vector<int, plato::mmap_allocator_t<non_trivial_t>> vec;

  for (int i = 0; i < 1033; ++i) {
    vec.emplace_back(i);
  }
  ASSERT_EQ(vec.size(), 1033);

  for (size_t i = 0; i < vec.size(); ++i) {
    ASSERT_THAT(vec, testing::Contains((int)i));
  }
}

