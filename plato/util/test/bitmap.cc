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

#include "plato/util/bitmap.hpp"

#include <mutex>
#include <vector>

#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(Bitmap, MSB) {
  plato::bitmap_t<> bitmap(40988);

  bitmap.set_bit(0);
  ASSERT_EQ(0, bitmap.msb());

  bitmap.set_bit(13);
  ASSERT_EQ(13, bitmap.msb());

  bitmap.set_bit(1982);
  ASSERT_EQ(1982, bitmap.msb());

  bitmap.clear();
  bitmap.set_bit(40987);
  ASSERT_EQ(40987, bitmap.msb());

  bitmap.set_bit(0);
  bitmap.set_bit(40986);
  ASSERT_EQ(40987, bitmap.msb());
}

TEST(Bitmap, TravseOrigin) {
  plato::bitmap_t<> bitmap(40988);
  bitmap.set_bit(0);
  bitmap.set_bit(19);
  bitmap.set_bit(40987);

  size_t chunk_size = 2;
  std::vector<plato::vid_t> vids;

  plato::traverse_opts_t opts;
  opts.mode_ = plato::traverse_mode_t::ORIGIN;

  bitmap.reset_traversal(opts);
  while (bitmap.next_chunk([&](plato::vid_t v_i) {
    vids.emplace_back(v_i);
  }, &chunk_size)) { }

  ASSERT_EQ(3, vids.size());

  ASSERT_EQ(0,     vids[0]);
  ASSERT_EQ(19,    vids[1]);
  ASSERT_EQ(40987, vids[2]);
}

TEST(Bitmap, TravseRandom) {
  plato::bitmap_t<> bitmap(40988);
  bitmap.set_bit(0);
  bitmap.set_bit(19);
  bitmap.set_bit(40987);

  std::vector<plato::vid_t> vids;

  plato::traverse_opts_t opts;
  opts.mode_ = plato::traverse_mode_t::RANDOM;

  bitmap.reset_traversal(opts);

  #pragma omp parallel
  {
    size_t chunk_size = 2;
    while (bitmap.next_chunk([&](plato::vid_t v_i) {
      #pragma omp critical
      vids.emplace_back(v_i);
    }, &chunk_size)) { }
  }

  ASSERT_EQ(3, vids.size());

  ASSERT_THAT(vids, testing::Contains(0));
  ASSERT_THAT(vids, testing::Contains(19));
  ASSERT_THAT(vids, testing::Contains(40987));
}

