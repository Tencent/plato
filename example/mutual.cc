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
#include <type_traits>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "boost/align.hpp"
#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/device/file.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/hdfs.hpp"
#include "plato/graph/base.hpp"
#include "plato/graph/state.hpp"
#include "plato/graph/structure.hpp"
#include "plato/graph/message_passing.hpp"
#include "plato/util/intersection.hpp"
#include "plato/algo/mutual/mutual.hpp"

DEFINE_string(input_edges,     "",         "input edges file, in csv format");
DEFINE_string(input_vertices,  "",         "input vertices file, in csv format");
DEFINE_string(output,          "",         "output directory");
DEFINE_string(separator,       ":",        "vertex states separator");
DEFINE_bool(common,            false,      "true: common, false: friends");
DEFINE_int32(vdata_bits,       32,         "vertex state data_bits: 16/32/64");
DEFINE_bool(ouput_list,        false,      "true: output common list, false: output common count.");

/**
 * @brief not empty validator.
 * @param value
 * @return
 */
bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

DEFINE_validator(input_edges,  &string_not_empty);
DEFINE_validator(output,  &string_not_empty);
DEFINE_validator(separator,  [] (const char*, const std::string& value) {return value.size() == 1;});
DEFINE_validator(vdata_bits, [] (const char*, const int value) {return value == 16 || value == 32 || value == 64;});

/**
 * @brief init
 * @param argc
 * @param argv
 */
void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

using edge_t = plato::empty_t;

template <typename T>
struct v_data_common_t {
  std::vector<T> mutual_;

  template<typename Ar>
  void serialize(Ar &ar) {
    ar & mutual_;
  }
};

template <typename T, bool from_mutual = true>
struct neighbors_trait {
  /**
   * @brief
   * @tparam GDATA
   * @param data
   * @param begin
   * @param end
   */
  template <typename GDATA>
  void operator() (GDATA& data, T** begin, T** end) {
    *begin = data.data_.mutual_.data();
    *end = *begin + data.data_.mutual_.size();
  }
};

template<>
struct neighbors_trait<plato::vid_t, false> {
  /**
   * @brief
   * @tparam GDATA
   * @param data
   * @param begin
   * @param end
   */
  template <typename GDATA>
  void operator() (GDATA& data, plato::vid_t** begin, plato::vid_t** end) {
    *begin = (plato::vid_t*)data.adjs_;
    *end = *begin + data.adjs_size_;
  }
};

/**
 * @brief load from file storage.
 * @tparam T
 * @tparam TCSR
 * @param tcsr
 */
template <typename T, typename TCSR>
void load_vertex_data(TCSR& tcsr, std::true_type) {
  using VDATA = typename TCSR::u_data_t;

  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::stop_watch_t watch;
  watch.mark("t1");

  auto pvcache = plato::load_vertices_cache<std::vector<T>>(
    FLAGS_input_vertices, plato::edge_format_t::CSV, [&](std::vector<T>* item, char* content) {
      const char* sep = FLAGS_separator.c_str();
      char* pSave  = nullptr;
      char* pToken = nullptr;
      pToken = strtok_r(content, sep, &pSave);
      while (pToken) {
        T val = std::strtoul(pToken, nullptr, 0);
        item->emplace_back(val);
        pToken = strtok_r(nullptr, sep, &pSave);
      }
      return true;
    });

  tcsr.template load_vertices_data_from_cache<std::vector<T>>(
    *pvcache,
    [](plato::vid_t /* v_i */, VDATA& v_data, std::vector<T>& value) {
      std::copy(value.begin(), value.end(), std::back_inserter(v_data.mutual_));
    }
  );

  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "load tcsr vertices costs: " << watch.show("t1") / 1000.0 << "s";
}

/**
 * @brief
 * @tparam T
 * @tparam TCSR
 */
template <typename T, typename TCSR>
void load_vertex_data(TCSR&, std::false_type) { }

/**
 * @brief calculate.
 * @tparam T
 * @tparam VDATA
 * @tparam from_mutual
 */
template <typename T, typename VDATA, bool from_mutual>
void process_mutual(void) {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::graph_info_t graph_info(false);

  plato::stop_watch_t watch;
  watch.mark("t0");
  watch.mark("t1");

  auto ptcsr = plato::create_tcsr_hashs_from_path<edge_t, VDATA>(
    &graph_info, FLAGS_input_edges, plato::edge_format_t::CSV, plato::dummy_decoder<edge_t>);
  auto& tcsr = *ptcsr;

  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "load tcsr edges costs: " << watch.show("t1") / 1000.0 << "s";

  load_vertex_data<T>(tcsr, std::integral_constant<bool, from_mutual>());

  {
    plato::thread_local_fs_output os(FLAGS_output, (boost::format("%04d_") % cluster_info.partition_id_).str(), true);

    plato::bsp_opts_t bsp_opts;
    bsp_opts.threads_               = -1;
    bsp_opts.flying_send_per_node_  = 1;
    bsp_opts.flying_recv_           = std::min(cluster_info.partitions_, cluster_info.threads_);
    bsp_opts.global_size_           = 16 * MBYTES;
    bsp_opts.local_capacity_        = 1024;
    bsp_opts.batch_size_            = 1;

    plato::mutual<T>(
      tcsr,
      neighbors_trait<T, from_mutual>(),
      [&](plato::vid_t src, plato::vid_t dst, T* mutual, size_t size_out) {
        auto& local_os = os.local();
        local_os << src << "," << dst << ",";
        if (FLAGS_ouput_list) {
          if (size_out) {
            for (size_t i = 0; i < size_out - 1; i++) {
              local_os << mutual[i] << ":";
            }
            local_os << mutual[size_out - 1];
          }
        } else {
          local_os << size_out;
        }
        local_os << std::endl;
      },
      bsp_opts);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "whole cost: " << watch.show("t0") / 1000.0 << "s";
}

int main(int argc, char** argv) {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "input_edges:     " << FLAGS_input_edges;
    LOG(INFO) << "input_vertices:  " << FLAGS_input_vertices;
    LOG(INFO) << "output:          " << FLAGS_output;
    LOG(INFO) << "separator:       " << FLAGS_separator;
    LOG(INFO) << "common:          " << FLAGS_common;
    LOG(INFO) << "vdata_bits:      " << FLAGS_vdata_bits;
  }

  if (FLAGS_common) {
    switch (FLAGS_vdata_bits) {
      case 16:
      {
        using T = uint16_t;
        using v_data_t = v_data_common_t<T>;
        process_mutual<T, v_data_t, true>();
        break;
      }
      case 32:
      {
        using T = uint32_t;
        using v_data_t = v_data_common_t<T>;
        process_mutual<T, v_data_t, true>();
        break;
      }
      case 64:
      {
        using T = uint64_t;
        using v_data_t = v_data_common_t<T>;
        process_mutual<T, v_data_t, true>();
        break;
      }
      default:
        abort();
    }
  } else {
    process_mutual<plato::vid_t, plato::empty_t, false>();
  }

  return 0;
}

