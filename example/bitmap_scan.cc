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

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "plato/util/perf.hpp"
#include "plato/util/bitmap.hpp"

DEFINE_uint32(size, 1000000, "bitmap's size");
DEFINE_uint32(seed, 777,     "rand seed");
DEFINE_double(load, 0.85,    "non-zero element ratio of the bitmap, [0, 1)");
DEFINE_bool(quick,  true,    "use quick scan or not");

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

int main(int argc, char** argv) {
  init(argc, argv);

  uint32_t nonzeros = (uint32_t)((double)FLAGS_size * FLAGS_load);
  if (nonzeros > FLAGS_size) { nonzeros = FLAGS_size; }

  LOG(INFO) << "generate random non-zero slot";

  plato::bitmap_t<> bitmap(FLAGS_size);

  if (nonzeros == FLAGS_size) {
    bitmap.fill();
  } else {
    srand(FLAGS_seed);

    uint32_t remain_elements = nonzeros;
    for (uint32_t i = 0; i < FLAGS_size; ++i) {
      if (((uint32_t)rand() % (FLAGS_size - i)) < remain_elements) {
        bitmap.set_bit(i);
        --remain_elements;
      }
    }
  }

  LOG(INFO) << "start scan bitmap with count: " << bitmap.count();

  plato::stop_watch_t watch;
  watch.mark("t0");

  uint32_t dummy = 0;
  if (FLAGS_quick) {
    size_t chunk_size = 64;
    bitmap.reset_traversal();

    #pragma omp parallel num_threads(24) reduction(+:dummy)
    {
      uint32_t __dummy = 0;
      while (bitmap.next_chunk([&](plato::vid_t v_i) {
        __dummy += 1;
      }, &chunk_size)) { }
      dummy += __dummy;
    }
  } else {
    for (uint32_t i = 0; i < FLAGS_size; ++i) {
      if (bitmap.get_bit(i)) {
        dummy += 1;
      }
    }
  }
  LOG(INFO) << "dummy: " << dummy;

  LOG(INFO) << "scan bitmap cost: " << watch.show("t0") / 1000.0 << "s";

  return 0;
}

