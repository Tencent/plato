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

#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <string.h>

#include "boost/filesystem.hpp"
#include "boost/format.hpp"
#include "glog/logging.h"

namespace plato {

class temporary_file_t {
  int fd_ = -1;
public:
  temporary_file_t(const temporary_file_t&) noexcept = delete;
  temporary_file_t& operator=(const temporary_file_t&) noexcept = delete;
  temporary_file_t(temporary_file_t&& x) noexcept : fd_(x.fd_) { x.fd_ = -1; }
  temporary_file_t& operator=(temporary_file_t&& x) noexcept {
    if (this != &x) {
      this->~temporary_file_t();
      new (this) temporary_file_t(std::move(x));
    }
    return *this;
  }

  temporary_file_t(std::string cache_dir = ".cache/") {
    if (!boost::filesystem::exists(cache_dir)) {
      boost::filesystem::create_directories(cache_dir);
    }
    CHECK(boost::filesystem::is_directory(cache_dir));

    std::string tmp_name = (boost::format("%s/XXXXXX") % cache_dir).str();
    fd_ = mkstemp64(const_cast<char*>(tmp_name.c_str()));
    CHECK(-1 != fd_) << boost::format("WARNING: mkstemp failed, err code: %d, err msg: %s") % errno % strerror(errno);
    CHECK(-1 != unlink(tmp_name.c_str())) << boost::format("WARNING: unlink failed, err code: %d, err msg: %s") % errno % strerror(errno);
  }

  ~temporary_file_t() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

  int fd() {
    CHECK(fd_ != -1);
    return fd_;
  }
};

}
