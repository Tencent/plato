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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "plato/util/perf.hpp"
#include "plato/util/aliastable.hpp"

DEFINE_uint32(size,  200,       "cardinal number");
DEFINE_uint32(count, 1000000,   "sample count");
DEFINE_uint32(seed,  777,       "rand seed");
DEFINE_string(type,  "native",  "'native' or 'alias'");
DEFINE_bool(prob,    true,      "visit probs array with sample index or not");

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

int main(int argc, char** argv) {
  init(argc, argv);

  std::mt19937 g1(FLAGS_seed);
  std::minstd_rand0 g2(FLAGS_seed);
  std::uniform_real_distribution<float> dist(0, 1.0);
  std::vector<float> probs(FLAGS_size);

  plato::stop_watch_t watch;

  for (size_t i = 0; i < FLAGS_size; ++i) {
    probs[i] = dist(g1);
  }

  if ("native" == FLAGS_type) {
    LOG(INFO) << "run native sample, preparing...";
    watch.mark("t1");

    double   dummy    = 0.0;
    uint64_t sampling = 0;

    std::vector<float> native_probs(probs.size());

    native_probs[0] = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) {
      native_probs[i] = probs[i] + native_probs[i - 1];
    }

    std::uniform_real_distribution<float> dist2(0, native_probs.back());
    LOG(INFO) << "preparing cost: " << watch.show("t1") / 1000.0 << "s";

    watch.mark("t1");

    if (FLAGS_prob) {
      while (sampling++ < FLAGS_count) {
        float next = dist2(g2);
        auto it = std::lower_bound(native_probs.begin(), native_probs.end(), next);
        dummy += probs[(it - native_probs.begin())];
      }
    } else {
      while (sampling++ < FLAGS_count) {
        float next = dist2(g2);
        auto it = std::lower_bound(native_probs.begin(), native_probs.end(), next);
        dummy += (double)(it - native_probs.begin());
      }
    }

    LOG(INFO) << "sampling [" << dummy << "/" << sampling - 1 << "] done, cost: "
      << watch.show("t1") / 1000.0 << "s";
  } else {
    LOG(INFO) << "run alias sample, preparing...";
    watch.mark("t1");

    double   dummy    = 0.0;
    uint64_t sampling = 0;

    plato::alias_table_t<float> alias(probs.data(), probs.size());

    if (FLAGS_prob) {
      LOG(INFO) << "preparing cost: " << watch.show("t1") / 1000.0 << "s"
        << ", run sample with visit probs";
      watch.mark("t1");
      while (sampling++ < FLAGS_count) {
        dummy += probs[alias.sample(g2)];
      }
    } else {
      LOG(INFO) << "preparing cost: " << watch.show("t1") / 1000.0 << "s"
        << ", run sample without visit probs";
      watch.mark("t1");
      while (sampling++ < FLAGS_count) {
        dummy += (double)alias.sample(g2);
      }
    }

    LOG(INFO) << "sampling [" << dummy << "/" << sampling - 1 << "] done, cost: "
      << watch.show("t1") / 1000.0 << "s";
  }

  return 0;
}

