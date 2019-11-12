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

#include <cstdint>
#include <cstdlib>
#include <functional>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "plato/util/perf.hpp"

DEFINE_uint64(times, 1000000, "bitmap's size");
DEFINE_string(type,  "all",   "'inline', 'lambda', 'function' or 'all'");

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

static uint64_t s_count = 0;

inline void plus_inline(uint64_t inc) {
  s_count += inc;
}

int main(int argc, char** argv) {
  init(argc, argv);

  auto plus_lambda = [&](int inc) {
    s_count += inc;
  };

  plato::stop_watch_t watch;
  watch.mark("t0");

  if ("all" == FLAGS_type || "inline" == FLAGS_type) {
    watch.mark("t1");
    s_count = 0;
    srand(777);
    for (uint64_t i = 0; i < FLAGS_times; ++i) {
      plus_inline(rand() % 10);
    }
    LOG(INFO) << "s_count: " << s_count << ", plus_inline cost: " << watch.show("t1") / 1000.0 << "s";
  }

  if ("all" == FLAGS_type || "lambda" == FLAGS_type) {
    watch.mark("t1");
    s_count = 0;
    srand(777);
    for (uint64_t i = 0; i < FLAGS_times; ++i) {
      plus_lambda(rand() % 10);
    }
    LOG(INFO) << "s_count: " << s_count << ", plus_lambda cost: " << watch.show("t1") / 1000.0 << "s";
  }

  if ("all" == FLAGS_type || "lambda" == FLAGS_type) {
    std::function<void(uint64_t inc)> plus_function(plus_lambda);

    watch.mark("t1");
    s_count = 0;
    srand(777);
    for (uint64_t i = 0; i < FLAGS_times; ++i) {
      plus_function(rand() % 10);
    }
    LOG(INFO) << "s_count: " << s_count << ", plus_function cost: " << watch.show("t1") / 1000.0 << "s";
  }

  return 0;
}

