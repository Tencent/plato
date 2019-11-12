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

#include <errno.h>
#include <malloc.h>
#include <string.h>

#include <random>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "glog/logging.h"
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "boost/format.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/defer.hpp"
#include "plato/util/intersection.hpp"

#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

template <typename T, typename SIZE_T>
void gen_id_list(SIZE_T len, double skew_ratio, double selectivity, double density, T*& set_a, T*& set_b, SIZE_T& size_a, SIZE_T& size_b) {
  ASSERT_TRUE(skew_ratio >= 1);
  ASSERT_TRUE(selectivity <= 1);
  ASSERT_TRUE(density <= 1);
  ASSERT_TRUE(len > 0);

  size_a = len;
  size_b = len / skew_ratio;
  SIZE_T size_out = std::min(size_a, size_b) * selectivity;
  SIZE_T range = (size_a + size_b - size_out) / density;

  posix_memalign(reinterpret_cast<void**>(&set_a), 32, sizeof(T) * size_a);
  posix_memalign(reinterpret_cast<void**>(&set_b), 32, sizeof(T) * size_b);

  std::mt19937 mt;
  std::uniform_int_distribution<SIZE_T> dist(0, range);

  std::unordered_set<T> ele_set;

  int x;
  for (SIZE_T i = 0; i < size_a; ++i) {
    do {
      x = dist(mt);
    } while (ele_set.find(x) != ele_set.end());
    ele_set.insert(x);
    set_a[i] = x;
  }

  for (SIZE_T i = 0; i < size_b; ++i) {
    if (i < size_out) {
      set_b[i] = set_a[i];
    } else {
      do {
        x = dist(mt);
      } while (ele_set.find(x) != ele_set.end());
      ele_set.insert(x);
      set_b[i] = x;
    }
  }

  std::sort(set_a, set_a + size_a);
  std::sort(set_b, set_b + size_b);

  LOG(INFO) << boost::format("gen_id_list done size_a=%d, size_b=%d, size_out=%d, range=%d") % size_a % size_b % size_out % range;
}

DEFINE_int32(len,              111111,       "length of array a");
DEFINE_double(skew,            1,            "skew_ratio");
DEFINE_double(selectivity,     0.1,          "selectivity");
DEFINE_double(density,         0.01,         "density");

DEFINE_validator(skew, [] (const char*, double skew) {return skew >= 1;});
DEFINE_validator(selectivity, [] (const char*, double selectivity) {return selectivity <= 1;});
DEFINE_validator(density, [] (const char*, double density) {return density <= 1;});

template <typename T>
void test32_64() {
  int len = std::min(FLAGS_len, std::numeric_limits<int>::max() / 4);
  T *set_a = nullptr, *set_b = nullptr, *baseline_set_out = nullptr, *set_out = nullptr;
  int size_a = 0, size_b = 0, baseline_size_out = 0, size_out = 0;
  gen_id_list(len, FLAGS_skew, FLAGS_selectivity, FLAGS_density, set_a, set_b, size_a, size_b);
  posix_memalign((void**)&baseline_set_out, sizeof(T) * 8, sizeof(T) * std::min(size_a, size_b));
  posix_memalign((void**)&set_out, sizeof(T) * 8, sizeof(T) * std::min(size_a, size_b));

  auto free_defer = plato::defer([&] {
    free(set_a);
    free(set_b);
    free(baseline_set_out);
    free(set_out);
  });
  UNUSED(free_defer);

  double cost;
  double baseline_cost;
  plato::stop_watch_t watch;

  {
    watch.mark("t0");
    baseline_size_out = plato::intersect_scalar(set_a, size_a, set_b, size_b, baseline_set_out);
    baseline_cost = watch.show("t0");
    LOG(INFO) << "\tscalar baseline\t" << baseline_cost;
  }

  {
    watch.mark("t1");
    size_out = plato::intersect_simd_shuffle(set_a, size_a, set_b, size_b, set_out);
    cost = watch.show("t1");
    LOG(INFO) << "\tsimd128 shuffling\t" << cost << "\tx" << baseline_cost / cost;
    ASSERT_TRUE(baseline_size_out == size_out);
    for (int i = 0; i < size_out; i++) {
      ASSERT_TRUE(set_out[i] == baseline_set_out[i]);
    }
  }

  {
    watch.mark("t1");
    size_out = plato::intersect_simd_shuffle_x2(set_a, size_a, set_b, size_b, set_out);
    cost = watch.show("t1");
    LOG(INFO) << "\tsimd128 shufflingx2\t" << cost << "\tx" << baseline_cost / cost;
    ASSERT_TRUE(baseline_size_out == size_out);
    for (int i = 0; i < size_out; i++) {
      ASSERT_TRUE(set_out[i] == baseline_set_out[i]);
    }
  }

  {
    watch.mark("t1");
    size_out = plato::intersect_simd_galloping(set_a, size_a, set_b, size_b, set_out);
    cost = watch.show("t1");
    LOG(INFO) << "\tsimd128 galloping\t" << cost << "\tx" << baseline_cost / cost;
    ASSERT_TRUE(baseline_size_out == size_out);
    for (int i = 0; i < baseline_size_out; i++) {
      ASSERT_TRUE(set_out[i] == baseline_set_out[i]);
    }
  }

  {
    watch.mark("t1");
    size_out = plato::intersect_simd(set_a, size_a, set_b, size_b, set_out);
    cost = watch.show("t1");
    LOG(INFO) << "\tsimd128 intersect\t" << cost << "\tx" << baseline_cost / cost;
    ASSERT_TRUE(baseline_size_out == size_out);
    for (int i = 0; i < size_out; i++) {
      ASSERT_TRUE(set_out[i] == baseline_set_out[i]);
    }
  }

#ifdef __AVX2__
  {
    watch.mark("t1");
    size_out = plato::intersect_simd_shuffle_avx(set_a, size_a, set_b, size_b, set_out);
    cost = watch.show("t1");
    LOG(INFO) << "\tsimd256 shuffling\t" << cost << "\tx" << baseline_cost / cost;
    ASSERT_TRUE(baseline_size_out == size_out);
    for (int i = 0; i < size_out; i++) {
      ASSERT_TRUE(set_out[i] == baseline_set_out[i]);
    }
  }

  {
    watch.mark("t1");
    size_out = plato::intersect_simd_shuffle_avx_x2(set_a, size_a, set_b, size_b, set_out);
    cost = watch.show("t1");
    LOG(INFO) << "\tsimd256 shufflingx2\t" << cost << "\tx" << baseline_cost / cost;
    ASSERT_TRUE(baseline_size_out == size_out);
    for (int i = 0; i < size_out; i++) {
      ASSERT_TRUE(set_out[i] == baseline_set_out[i]);
    }
  }

  {
    watch.mark("t1");
    size_out = plato::intersect_simd_galloping_avx(set_a, size_a, set_b, size_b, set_out);
    cost = watch.show("t1");
    LOG(INFO) << "\tsimd256 galloping\t" << cost << "\tx" << baseline_cost / cost;
    ASSERT_TRUE(baseline_size_out == size_out);
    for (int i = 0; i < size_out; i++) {
      ASSERT_TRUE(set_out[i] == baseline_set_out[i]);
    }
  }

  {
    watch.mark("t1");
    size_out = plato::intersect_simd_avx(set_a, size_a, set_b, size_b, set_out);
    cost = watch.show("t1");
    LOG(INFO) << "\tsimd256 intersect\t" << cost << "\tx" << baseline_cost / cost;
    ASSERT_TRUE(baseline_size_out == size_out);
    for (int i = 0; i < size_out; i++) {
      ASSERT_TRUE(set_out[i] == baseline_set_out[i]);
    }
  }
#endif
}

void test16() {
  int len = std::min(FLAGS_len, (int)(std::numeric_limits<uint16_t >::max() / 4));
  uint16_t *set_a = nullptr, *set_b = nullptr, *baseline_set_out = nullptr, *set_out = nullptr;
  int size_a = 0, size_b = 0, baseline_size_out = 0, size_out = 0;
  gen_id_list(len, FLAGS_skew, FLAGS_selectivity, FLAGS_density, set_a, set_b, size_a, size_b);
  posix_memalign((void**)&baseline_set_out, sizeof(uint16_t) * 8, sizeof(uint16_t) * std::min(size_a, size_b));
  posix_memalign((void**)&set_out, sizeof(uint16_t) * 8, sizeof(uint16_t) * std::min(size_a, size_b));

  auto free_defer = plato::defer([&] {
    free(set_a);
    free(set_b);
    free(baseline_set_out);
    free(set_out);
  });
  UNUSED(free_defer);

  double cost;
  double baseline_cost;
  plato::stop_watch_t watch;

  {
    watch.mark("t1");
    baseline_size_out = plato::intersect_scalar(set_a, size_a, set_b, size_b, baseline_set_out);
    baseline_cost = watch.show("t1");
    LOG(INFO) << "\tscalar baseline\t" << baseline_cost;
  }

  {
    watch.mark("t1");
    size_out = plato::intersect_simd_sttni(set_a, size_a, set_b, size_b, set_out);
    cost = watch.show("t1");
    LOG(INFO) << "\tsimd128 sttni\t" << cost << "\tx" << baseline_cost / cost;
    ASSERT_TRUE(baseline_size_out == size_out);
    for (int i = 0; i < size_out; i++) {
      ASSERT_TRUE(set_out[i] == baseline_set_out[i]);
    }
  }

  {
    watch.mark("t1");
    size_out = plato::intersect_simd_sttni_x2(set_a, size_a, set_b, size_b, set_out);
    cost = watch.show("t1");
    LOG(INFO) << "\tsimd128 sttnix2\t" << cost << "\tx" << baseline_cost / cost;
    ASSERT_TRUE(baseline_size_out == size_out);
    for (int i = 0; i < size_out; i++) {
      ASSERT_TRUE(set_out[i] == baseline_set_out[i]);
    }
  }
}

TEST(intersection, 32bit) {
  test32_64<uint32_t >();
}

TEST(intersection, 64bit) {
  test32_64<uint64_t >();
}

TEST(intersection, 16bit) {
  test16();
}

int main(int argc, char* argv[]) {
  // Filter out Google Test arguments
  ::testing::InitGoogleTest(&argc, argv);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging("intersection-test");
  google::LogToStderr();

  // Run tests, then clean up and exit
  return RUN_ALL_TESTS();
}

