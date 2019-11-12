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

#include "plato/util/aliastable.hpp"

#include <cmath>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(AliasMethod, FloatSampling) {
  std::vector<float> probs({ 0.1, 0.2, 0.5, 0.2 });
  plato::alias_table_t<float> alias(probs.data(), probs.size());

  for (size_t i = 1; i < probs.size(); ++i) {
    probs[i] = probs[i] + probs[i - 1];
  }

  std::mt19937 g1(777);
  std::uniform_real_distribution<float> dist(0, probs.back());
  const size_t total_count = 10000000;

  std::vector<size_t> pre_counts(probs.size(), 0);
  for (size_t i = 0; i < total_count; ++i) {
    float next = dist(g1);
    auto it = std::lower_bound(probs.begin(), probs.end(), next);
    pre_counts[it - probs.begin()]++;
  }
  for (size_t i = 0; i < pre_counts.size(); ++i) {
    fprintf(stderr, "native: %lu,%lu\n", i, pre_counts[i]);
  }

  std::mt19937 g2(777);
  std::vector<size_t> alias_counts(probs.size(), 0);
  for (size_t i = 0; i < total_count; ++i) {
    alias_counts[alias.sample(g2)]++;
  }
  for (size_t i = 0; i < pre_counts.size(); ++i) {
    fprintf(stderr, "alias: %lu,%lu\n", i, alias_counts[i]);
  }

  for (size_t i = 0; i < probs.size(); ++i) {
    ASSERT_LT(fabs(((float)pre_counts[i] - (float)alias_counts[i]) / (float)pre_counts[i]), 1e-2);
  }
}

TEST(AliasMethod, DoubleSampling) {
  std::vector<double> probs({ 0.1, 0.2, 0.5, 0.2, 1.0 });
  plato::alias_table_t<double> alias(probs.data(), probs.size());

  for (size_t i = 1; i < probs.size(); ++i) {
    probs[i] = probs[i] + probs[i - 1];
  }

  std::mt19937 g1(777);
  std::uniform_real_distribution<double> dist(0, probs.back());
  const size_t total_count = 10000000;

  std::vector<size_t> pre_counts(probs.size(), 0);
  for (size_t i = 0; i < total_count; ++i) {
    double next = dist(g1);
    auto it = std::lower_bound(probs.begin(), probs.end(), next);
    pre_counts[it - probs.begin()]++;
  }
  for (size_t i = 0; i < pre_counts.size(); ++i) {
    fprintf(stderr, "native: %lu,%lu\n", i, pre_counts[i]);
  }

  std::mt19937 g2(777);
  std::vector<size_t> alias_counts(probs.size(), 0);
  for (size_t i = 0; i < total_count; ++i) {
    alias_counts[alias.sample(g2)]++;
  }
  for (size_t i = 0; i < pre_counts.size(); ++i) {
    fprintf(stderr, "alias: %lu,%lu\n", i, alias_counts[i]);
  }

  for (size_t i = 0; i < probs.size(); ++i) {
    ASSERT_LT(fabs(((double)pre_counts[i] - (double)alias_counts[i]) / (double)pre_counts[i]), 1e-2);
  }
}

TEST(AliasMethod, InitInplace) {
  std::vector<double> probs({ 0.1, 0.2, 0.5, 0.2, 1.0 });
  plato::alias_table_t<double> alias(probs.size());

  for (size_t i = 0; i < probs.size(); ++i) {
    alias[i].prob_ = probs[i];
  }
  alias.initialize();

  for (size_t i = 1; i < probs.size(); ++i) {
    probs[i] = probs[i] + probs[i - 1];
  }

  std::mt19937 g1(777);
  std::uniform_real_distribution<double> dist(0, probs.back());
  const size_t total_count = 10000000;

  std::vector<size_t> pre_counts(probs.size(), 0);
  for (size_t i = 0; i < total_count; ++i) {
    double next = dist(g1);
    auto it = std::lower_bound(probs.begin(), probs.end(), next);
    pre_counts[it - probs.begin()]++;
  }
  for (size_t i = 0; i < pre_counts.size(); ++i) {
    fprintf(stderr, "native: %lu,%lu\n", i, pre_counts[i]);
  }

  std::mt19937 g2(777);
  std::vector<size_t> alias_counts(probs.size(), 0);
  for (size_t i = 0; i < total_count; ++i) {
    alias_counts[alias.sample(g2)]++;
  }
  for (size_t i = 0; i < pre_counts.size(); ++i) {
    fprintf(stderr, "alias: %lu,%lu\n", i, alias_counts[i]);
  }

  for (size_t i = 0; i < probs.size(); ++i) {
    ASSERT_LT(fabs(((double)pre_counts[i] - (double)alias_counts[i]) / (double)pre_counts[i]), 1e-2);
  }
}

TEST(AliasMethod, Resize) {
  std::vector<double> probs({ 0.1, 0.2, 0.5, 0.2, 1.0 });
  plato::alias_table_t<double> alias(3);

  for (size_t i = 0; i < 3; ++i) {
    alias[i].prob_ = probs[i];
  }

  alias.resize(probs.size() + 10);
  for (size_t i = 3; i < probs.size(); ++i) {
    alias[i].prob_ = probs[i];
  }

  alias.resize(probs.size());
  alias.initialize();

  for (size_t i = 1; i < probs.size(); ++i) {
    probs[i] = probs[i] + probs[i - 1];
  }

  std::mt19937 g1(777);
  std::uniform_real_distribution<double> dist(0, probs.back());
  const size_t total_count = 10000000;

  std::vector<size_t> pre_counts(probs.size(), 0);
  for (size_t i = 0; i < total_count; ++i) {
    double next = dist(g1);
    auto it = std::lower_bound(probs.begin(), probs.end(), next);
    pre_counts[it - probs.begin()]++;
  }
  for (size_t i = 0; i < pre_counts.size(); ++i) {
    fprintf(stderr, "native: %lu,%lu\n", i, pre_counts[i]);
  }

  std::mt19937 g2(777);
  std::vector<size_t> alias_counts(probs.size(), 0);
  for (size_t i = 0; i < total_count; ++i) {
    alias_counts[alias.sample(g2)]++;
  }
  for (size_t i = 0; i < pre_counts.size(); ++i) {
    fprintf(stderr, "alias: %lu,%lu\n", i, alias_counts[i]);
  }

  for (size_t i = 0; i < probs.size(); ++i) {
    ASSERT_LT(fabs(((double)pre_counts[i] - (double)alias_counts[i]) / (double)pre_counts[i]), 1e-2);
  }
}

