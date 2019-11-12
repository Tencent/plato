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
#include <vector>
#include <memory>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "plato/util/perf.hpp"
#include "plato/util/archive.hpp"
#include "yas/types/std/vector.hpp"

DEFINE_uint64(count,      1000000, "sample count");
DEFINE_uint32(thread,     4,       "threads number");
DEFINE_uint32(vsize,      100,     "vector size for non-trivial type");
DEFINE_bool(is_trivial,   false,   "trivial or not");

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

int main(int argc, char** argv) {
  init(argc, argv);

  LOG(INFO) << "count:   " << FLAGS_count;
  LOG(INFO) << "threads: " << FLAGS_thread;

  if (false == FLAGS_is_trivial) {
    using oarchive_spec_t = plato::oarchive_t<std::vector<uint32_t>, plato::mem_ostream_t>;
    using iarchive_spec_t = plato::iarchive_t<std::vector<uint32_t>, plato::mem_istream_t>;

    std::vector<std::unique_ptr<oarchive_spec_t>> oarchive_vec(FLAGS_thread);
    std::vector<std::unique_ptr<iarchive_spec_t>> iarchive_vec(FLAGS_thread);
    std::vector<size_t> obj_count(FLAGS_thread, 0);

    plato::stop_watch_t watch;
    watch.mark("t1");
    #pragma omp parallel num_threads(FLAGS_thread)
    {
      oarchive_vec[omp_get_thread_num()].reset(new oarchive_spec_t);
      auto&   oarchive = *oarchive_vec[omp_get_thread_num()];
      size_t& count    = obj_count[omp_get_thread_num()];

      std::vector<uint32_t> output(FLAGS_vsize);
      memset(output.data(), 0xaa, sizeof(uint32_t) * FLAGS_vsize);
      #pragma omp for
      for (size_t i = 0; i < FLAGS_count; ++i) {
        oarchive.emit(output);
        ++count;
      }
    }
    LOG(INFO) << "serialize cost: " << watch.show("t1") / 1000.0 << "s";

    watch.mark("t1");
    #pragma omp parallel num_threads(FLAGS_thread)
    {
      auto buff = oarchive_vec[omp_get_thread_num()]->get_intrusive_buffer();
      iarchive_vec[omp_get_thread_num()].reset(new iarchive_spec_t(buff.data_, buff.size_,
            obj_count[omp_get_thread_num()]));
      auto& iarchive = *iarchive_vec[omp_get_thread_num()];

      CHECK(false == iarchive.is_trivial());

      #pragma omp for
      for (size_t i = 0; i < FLAGS_count; ++i) {
        auto __tmp = iarchive.absorb();
        CHECK(__tmp->size() == FLAGS_vsize);
      }
    }
    LOG(INFO) << "deserialize cost: " << watch.show("t1") / 1000.0 << "s";

  } else {
    CHECK(false) << "trivial type is not implemented";
  }

  return 0;
}

