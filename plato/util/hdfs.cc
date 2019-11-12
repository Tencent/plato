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

#include "hdfs.hpp"

#include "omp.h"

#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"
#include "boost/iostreams/device/file.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/join.hpp"

namespace plato {

std::mutex hdfs_t::mtx4inst_;
std::map<std::string, std::shared_ptr<hdfs_t>> hdfs_t::instances_;

std::string hdfs_t::get_nm_from_path(const std::string& path) {
  if (boost::istarts_with(path, "hdfs://")) {
    std::vector<std::string> splits;
    boost::split(splits, path, boost::is_any_of("/"));

    CHECK(splits.size() >= 3) << splits.size() << ", " << path;
    return splits[2];
  } else {
    return "default";
  }
}

hdfs_t& hdfs_t::get_hdfs(void) {
  std::lock_guard<std::mutex> lck(mtx4inst_);

  if (instances_.count("default")) {
    return *instances_["default"];
  } else {
    instances_.emplace("default", std::shared_ptr<hdfs_t>(new hdfs_t()));
  }
  return *instances_["default"];
}

hdfs_t& hdfs_t::get_hdfs(const std::string& path) {
  std::lock_guard<std::mutex> lck(mtx4inst_);

  std::string nm = get_nm_from_path(path);
  if (instances_.count(nm)) {
    return *instances_[nm];
  } else {
    instances_.emplace(nm, std::shared_ptr<hdfs_t>(new hdfs_t(nm)));
  }
  return *instances_[nm];
}

void hdfs_t::parse_csv_files(const hdfs_t& fs, const std::vector<std::string>& chunks,
    std::function<void(const std::vector<std::vector<std::string>>&)> callback) {
  UNUSED(fs);
  using STREAM_T = boost::iostreams::filtering_stream<boost::iostreams::input>;
  std::vector<hdfs_t::fstream*> chunk_stream;
  std::vector<STREAM_T*> fin;

  for (const auto& chunk: chunks) {
    chunk_stream.emplace_back(new hdfs_t::fstream(hdfs_t::get_hdfs(chunk), chunk));
    fin.emplace_back(new STREAM_T());

    if (boost::ends_with(chunk, ".gz")) {
      fin.back()->push(boost::iostreams::gzip_decompressor());
    }
    fin.back()->push(*chunk_stream.back());
  }

  #pragma omp parallel
  {
    std::vector<std::vector<std::string>> blocks;
    std::string oneline;

    #pragma omp for schedule(dynamic)
    for (size_t i = 0; i < fin.size(); ++i) {
      auto& sin = fin[i];
      while (sin->good() && (false == sin->eof())) {
        std::getline(*sin, oneline);

        if ((false == sin->good()) || (0 == oneline.length())) {
          continue;
        }

        std::vector<std::string> splits;
        boost::split(splits, oneline, boost::is_any_of(","));

        blocks.emplace_back(std::move(splits));

        if (blocks.size() > (1 << 20)) {
          callback(blocks);
          blocks.clear();
        }
      }
      if (blocks.size()) {
        callback(blocks);
        blocks.clear();
      }
    }
  }

  for (auto&f: fin) {
    delete f;
  }
  for (auto&s: chunk_stream) {
    delete s;
  }
  fin.clear();
  chunk_stream.clear();
}

}  // end of namespace plato

