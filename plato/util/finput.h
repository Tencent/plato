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

#ifndef __PLATO_UTIL_FINPUT_HPP__
#define __PLATO_UTIL_FINPUT_HPP__

#include "boost/format.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"
#include "boost/iostreams/device/file.hpp"

#include "plato/util/hdfs.hpp"

namespace plato {

/*
 * \param filename  file name
 * \param func      auto func(boost::iostreams::filtering_istream& os)
 *
 * \return 0 -- success, else -- file not existed or other error
 **/
template <typename Func>
inline int with_ifile(const std::string& filename, Func&& func) {
  boost::iostreams::filtering_istream fin;
  if (boost::iends_with(filename, ".gz")) {
    fin.push(boost::iostreams::gzip_decompressor());
  }

  if (boost::istarts_with(filename, "hdfs://")) {
    if (false == hdfs_t::get_hdfs(filename).exists(filename)) {
      return -1;
    }

    hdfs_t::fstream hdfs_fin(hdfs_t::get_hdfs(filename), filename);
    fin.push(hdfs_fin);
    func(fin);
  } else {
    if (false == boost::filesystem::exists(filename)) {
      return -1;
    }

    fin.push(boost::iostreams::file_source(filename));
    func(fin);
  }
  return 0;
}

}

#endif

