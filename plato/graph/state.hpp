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

#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <memory>
#include <functional>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "boost/format.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/graph/base.hpp"
#include "plato/graph/detail.hpp"
#include "plato/graph/parsers.hpp"
#include "plato/graph/state/view.hpp"
#include "plato/graph/state/vertex_cache.hpp"
#include "plato/graph/state/dense_state.hpp"
#include "plato/graph/state/sparse_state.hpp"
#include "plato/graph/state/allgather_state.hpp"
#include "plato/util/hdfs.hpp"
#include "plato/util/perf.hpp"
#include "plato/util/mmap_alloc.hpp"
#include "plato/parallel/mpi.hpp"
#include "plato/parallel/bsp.hpp"

#include "plato/util/bitmap.hpp"

namespace plato {

/*
 * parallel load vertices from file system to cache
 *
 * \tparam VDATA   vertex data type
 * \tparam VCACHE  cache type, can be 'vertex_cache_t' or 'vertex_file_cache_t'
 *
 * \param pginfo   graph info
 * \param path     input file path, 'path' can be a file or a directory.
 *                 'path' can be located on hdfs or posix, distinguish by its prefix.
 *                 eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format   file format
 * \param decoder  edge data decode, string => VDATA
 *
 * \return loaded cache or nullptr
 **/
template <typename VDATA, template<typename> class VCACHE = vertex_cache_t>
std::shared_ptr<VCACHE<VDATA>> load_vertices_cache(
  const std::string& path,
  edge_format_t      format,
  decoder_t<VDATA>   decoder) {

  std::shared_ptr<VCACHE<VDATA>> pcache(new VCACHE<VDATA>());
  auto& cluster_info = cluster_info_t::get_instance();

  auto callback = [&] (vertex_unit_t<VDATA>* input, size_t size) {
    pcache->push_back(input, size);
    return true;
  };

  vertex_parser_t<boost::iostreams::filtering_istream, VDATA> parser;
  switch (format) {
    case edge_format_t::CSV:
      parser = vertex_csv_parser<boost::iostreams::filtering_istream, VDATA>;
      break;
    default:
      LOG(ERROR) << "unknown format: " << (uint64_t)format;
      return nullptr;
  }

  std::vector<std::string> files = get_files(path);
  std::mutex files_lock;

  #pragma omp parallel num_threads(cluster_info.threads_)
  {
    while (true) {
      std::string filename;
      {
        std::lock_guard<std::mutex> lock(files_lock);
        if (files.empty()) break;
        filename = std::move(files.back());
        files.pop_back();
      }

      with_file(filename, [&] (boost::iostreams::filtering_istream& is) {
        parser(is, callback, decoder);
      });
    }
  }

  return pcache;
}

/*
 * parallel load vertices state from file system
 *
 * \tparam T_MSG       type used for passing state to its partition
 * \tparam PART_IMPL   partitioner's type
 * \tparam CALLBACK    callback functor to deal with received T_MSG, should implement the methods:
 *                     <tt>void(vertex_unit_t<T_MSG>&&)<\tt>
 *
 * \param path         input file path, 'path' can be a file or a directory.
 *                     'path' can be located on hdfs or posix, distinguish by its prefix.
 *                     eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format       file format
 * \param decoder      edge data decode, string => T_MSG
 * \param partitioner  vertex partitioner
 * \param callback     received states callback fuction
 **/
template <typename T_MSG, typename PART_IMPL, typename CALLBACK>
void load_vertices_state_from_path (
  const std::string&          path,
  edge_format_t               format,
  std::shared_ptr<PART_IMPL>  partitioner,
  decoder_t<T_MSG>            decoder,
  CALLBACK&&                  callback) {

  plato::stop_watch_t watch;
  auto& cluster_info = cluster_info_t::get_instance();

  vertex_parser_t<boost::iostreams::filtering_istream, T_MSG> parser;
  switch (format) {
  case edge_format_t::CSV:
    parser = vertex_csv_parser<boost::iostreams::filtering_istream, T_MSG>;
    break;
  default:
    CHECK(false) << "unknown format: " << (uint64_t)format;
  }

  watch.mark("t1");
  {
    std::atomic<size_t> idx(0);
    std::vector<std::string> files = get_files(path);

    auto __send = [&](bsp_send_callback_t<vertex_unit_t<T_MSG>> send) {
      auto parse_callback = [&](vertex_unit_t<T_MSG>* input, size_t size) {
        for (size_t i = 0; i < size; ++i) {
          send(partitioner->get_partition_id(input[i].vid_), input[i]);
        }
        return true;
      };

      size_t __idx = idx.fetch_add(1, std::memory_order_relaxed);
      while (__idx < files.size()) {
        with_file(files[__idx], [&](boost::iostreams::filtering_istream& is) {
          parser(is, parse_callback, decoder);
        });
        __idx = idx.fetch_add(1, std::memory_order_relaxed);
      }
    };

    auto __recv = [&](int, bsp_recv_pmsg_t<vertex_unit_t<T_MSG>>& pmsg) {
      callback(std::move(*pmsg));
    };

    bsp_opts_t bsp_opts;
    bsp_opts.global_size_    = 64 * MBYTES;
    bsp_opts.local_capacity_ = 32 * PAGESIZE;

    fine_grain_bsp<vertex_unit_t<T_MSG>>(__send, __recv, bsp_opts);
  }
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "load vertex state from path done, cost: " << watch.show("t1") / 1000.0 << "s";
  }
}

}

