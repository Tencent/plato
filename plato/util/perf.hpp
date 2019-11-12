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

#ifndef __PLATO_PERF_HPP__
#define __PLATO_PERF_HPP__

#include <sys/time.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <map>
#include <string>

namespace plato {

struct mem_status_t {
  size_t vm_peak; // KBytes
  size_t vm_size; // KBytes
  size_t vm_hwm;  // KBytes
  size_t vm_rss;  // KBytes
};

inline void self_mem_usage(mem_status_t* status) {
  char buffer[1024] = "";

  FILE* file = fopen("/proc/self/status", "r");
  while (fscanf(file, " %1023s", buffer) == 1) {
    if (strcmp(buffer, "VmRSS:") == 0) {
      fscanf(file, " %lu", &status->vm_rss);
    }
    if (strcmp(buffer, "VmHWM:") == 0) {
      fscanf(file, " %lu", &status->vm_hwm);
    }
    if (strcmp(buffer, "VmSize:") == 0) {
      fscanf(file, " %lu", &status->vm_size);
    }
    if (strcmp(buffer, "VmPeak:") == 0) {
      fscanf(file, " %lu", &status->vm_peak);
    }
  }
  fclose(file);
}

inline double current_milliseconds(void) {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000UL + tv.tv_usec / 1000UL;
}

class stop_watch_t {
public:
  // start a new mark
  void mark(const std::string& smark) {
    mark_[smark] = current_milliseconds();
  }

  // return a mark's cost in milliseconds
  double show(const std::string& smark) {
    return current_milliseconds() - mark_[smark];
  }

  // return a mark's cost in milliseconds
  std::string showlit_mills(const std::string& smark) {
    double cost = current_milliseconds() - mark_[smark];
    return std::string(std::to_string(cost) + "ms");
  }

  // return a mark's cost in milliseconds
  std::string showlit_seconds(const std::string& smark) {
    double cost = current_milliseconds() - mark_[smark];
    return std::string(std::to_string(cost / 1000.0) + "s");
  }

  // remove mark and return its cost
  double stop(const std::string& smark) {
    double cost = current_milliseconds() - mark_[smark];
    mark_.erase(smark);
    return cost;
  }

protected:
  std::map<std::string, double> mark_;
};

}  // namespace plato

#endif

