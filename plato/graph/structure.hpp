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

#ifndef __PLATO_GRAPH_STRUCTURE_HPP__
#define __PLATO_GRAPH_STRUCTURE_HPP__

#include <cstdint>
#include <cstdlib>

#include <list>
#include <vector>
#include <atomic>
#include <thread>
#include <limits>
#include <memory>
#include <utility>
#include <fstream>
#include <iostream>
#include <type_traits>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "boost/format.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/bitmap.hpp"
#include "plato/util/mmap_alloc.hpp"
#include "plato/parallel/mpi.hpp"
#include "plato/parallel/bsp.hpp"
#include "plato/graph/base.hpp"
#include "plato/graph/state.hpp"
#include "plato/graph/detail.hpp"
#include "plato/graph/parsers.hpp"
#include "plato/graph/structure/bcsr.hpp"
#include "plato/graph/structure/dcsc.hpp"
#include "plato/graph/structure/tcsr.hpp"
#include "plato/graph/structure/edge_cache.hpp"
#include "plato/graph/partition/hash.hpp"
#include "plato/graph/partition/dummy.hpp"
#include "plato/graph/partition/sequence.hpp"
#include "plato/graph/structure/vid_encoder.hpp"

#include "yas/types/std/vector.hpp"
#include "yas/types/std/string.hpp"

namespace plato {

// ******************************************************************************* //
// factory function for create edges

template <typename EDATA, typename VID_T = vid_t>
using data_callback_t = std::function<bool(edge_unit_t<EDATA, VID_T>*, size_t)>;

template <typename EDATA, typename VID_T = vid_t, template<typename, typename> class CACHE = edge_block_cache_t>
using vencoder_t = typename std::remove_reference<vid_encoder_t<EDATA,VID_T,CACHE>*>::type;

/*
 * parallel parse edges from file system to cache
 *
 * \tparam EDATA        data bind on edge
 * \tparam VID_T        vertex id type, can be uint32_t or uint64_t
 *
 * \param path          input file path, 'path' can be a file or a directory.
 *                      'path' can be located on hdfs or posix, distinguish by its prefix.
 *                      eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format        file format
 * \param decoder       edge data decode, string => EDATA
 * \param callback      function executed when parsing data
 *
 **/
template <typename EDATA, typename VID_T = vid_t>
void read_from_files(
  const std::string&            path,
  edge_format_t                 format,
  decoder_t<EDATA>              decoder,
  data_callback_t<EDATA, VID_T> callback) {

  auto& cluster_info = cluster_info_t::get_instance();
  edge_parser_t<boost::iostreams::filtering_istream, EDATA, VID_T> parser;
  switch (format) {
    case edge_format_t::CSV:
      parser = csv_parser<boost::iostreams::filtering_istream, EDATA, VID_T>;
      break;
    default:
      LOG(ERROR) << "unknown format: " << (uint64_t)format;
      throw std::runtime_error((boost::format("unknown format: %lu") % (uint64_t)format).str());
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

}

/*
 * parallel load edges with encoder from file system to cache
 *
 * \tparam EDATA        data bind on edge
 * \tparam VID_T        vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE        cache type, can be edge_block_cache_t or edge_file_cache_t or edge_cache_t
 *
 * \param path          input file path, 'path' can be a file or a directory.
 *                      'path' can be located on hdfs or posix, distinguish by its prefix.
 *                      eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format        file format
 * \param decoder       edge data decode, string => EDATA
 * \param callback      function executed when parsing data
 * \param vid_encoder   encoder for data 
 *
 **/
template <typename EDATA, typename VID_T = vid_t, template<typename, typename> class CACHE = edge_block_cache_t>
void load_edges_cache_with_encoder(
    const std::string&              path,
    edge_format_t                   format,
    decoder_t<EDATA>                decoder,
    data_callback_t<EDATA, vid_t>   callback,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr) {

  std::shared_ptr<CACHE<EDATA, VID_T>> pcache(new CACHE<EDATA, VID_T>());

  // we count every statistics first, do not optimized early
  auto read_callback = [&](edge_unit_t<EDATA, VID_T>* input, size_t size) {
    pcache->push_back(input, size);
    return true;
  };

  read_from_files<EDATA, VID_T>(path, format, decoder, read_callback); 
  vid_encoder->encode(*pcache, callback);
}

/*
 * parallel load edges from file system to cache
 *
 * \tparam EDATA        data bind on edge
 * \tparam VID_T        vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE        cache type, can be edge_block_cache_t or edge_file_cache_t or edge_cache_t
 *
 * \param pginfo        graph info
 * \param path          input file path, 'path' can be a file or a directory.
 *                      'path' can be located on hdfs or posix, distinguish by its prefix.
 *                      eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format        file format
 * \param decoder       edge data decode, string => EDATA
 * \param callback      function executed when parsing data
 * \param vid_encoder   encoder for data 
 *
 * \return loaded cache or nullptr
 **/
template <typename EDATA, typename VID_T = vid_t, template<typename, typename> class CACHE = edge_block_cache_t>
std::shared_ptr<CACHE<EDATA, vid_t>> load_edges_cache(
    graph_info_t*                   pginfo,
    const std::string&              path,
    edge_format_t                   format,
    decoder_t<EDATA>                decoder,
    data_callback_t<EDATA, vid_t>   callback = nullptr,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr) {

  eid_t edges = 0;
  bitmap_t<> v_bitmap(std::numeric_limits<vid_t>::max());
  std::shared_ptr<CACHE<EDATA, vid_t>> cache(new CACHE<EDATA, vid_t>());
  auto real_callback = [&](edge_unit_t<EDATA, vid_t>* input, size_t size) {
    __sync_fetch_and_add(&edges, size);
    for (size_t i = 0; i < size; ++i) {
      v_bitmap.set_bit(input[i].src_);
      v_bitmap.set_bit(input[i].dst_);
    }
    cache->push_back(input, size);
    if (nullptr != callback) callback(input, size);
    return true;
  };

  if (nullptr != vid_encoder) {
    load_edges_cache_with_encoder<EDATA, VID_T, CACHE>(path, format, 
        decoder, real_callback, vid_encoder);
  }
  else {
    bool is_uint64 = std::is_same<VID_T, uint64_t>::value;
    if (is_uint64) {
      LOG(ERROR) << "cannot read uint64 without vid encoder: " << (uint64_t)format;
      return nullptr;
    }
    read_from_files<EDATA, vid_t>(path, format, decoder, real_callback); 
  }

  MPI_Allreduce(MPI_IN_PLACE, &edges, 1, get_mpi_data_type<eid_t>(), MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, v_bitmap.data_, word_offset(v_bitmap.size_) + 1, get_mpi_data_type<uint64_t>(),
      MPI_BOR, MPI_COMM_WORLD);

  if (pginfo) {
    pginfo->edges_    = edges;
    pginfo->vertices_ = v_bitmap.count();
    pginfo->max_v_i_  = v_bitmap.msb();
  }

  return cache;
}

template <typename T, typename EDGE_CACHE>
std::vector<T> generate_dense_out_degrees(const graph_info_t& graph_info, EDGE_CACHE& cache) {
  using edge_unit_spec_t = typename EDGE_CACHE::edge_unit_spec_t;

  std::vector<T> degrees(graph_info.max_v_i_ + 1, 0);

  cache.reset_traversal();
  #pragma omp parallel
  {
    auto traversal = [&](size_t /*idx*/, edge_unit_spec_t* edge) {
      __sync_fetch_and_add(&degrees[edge->src_], 1);
      if (false == graph_info.is_directed_) {
        __sync_fetch_and_add(&degrees[edge->dst_], 1);
      }
      return true;
    };

    size_t chunk_size = 64;
    while (cache.next_chunk(traversal, &chunk_size)) { }
  }
  MPI_Allreduce(MPI_IN_PLACE, degrees.data(), graph_info.max_v_i_ + 1, get_mpi_data_type<T>(), MPI_SUM, MPI_COMM_WORLD);

  return degrees;
}

template <typename T, typename EDGE_CACHE>
std::vector<T> generate_dense_in_degrees(const graph_info_t& graph_info, EDGE_CACHE& cache) {
  using edge_unit_spec_t = typename EDGE_CACHE::edge_unit_spec_t;

  std::vector<T> degrees(graph_info.max_v_i_ + 1, 0);

  cache.reset_traversal();
  #pragma omp parallel
  {
    auto traversal = [&](size_t, edge_unit_spec_t* edge) {
      __sync_fetch_and_add(&degrees[edge->dst_], 1);
      if (false == graph_info.is_directed_) {
        __sync_fetch_and_add(&degrees[edge->src_], 1);
      }
      return true;
    };

    size_t chunk_size = 64;
    while (cache.next_chunk(traversal, &chunk_size)) { }
  }
  MPI_Allreduce(MPI_IN_PLACE, degrees.data(), graph_info.max_v_i_ + 1, get_mpi_data_type<T>(), MPI_SUM, MPI_COMM_WORLD);

  return degrees;
}

// generate dense out degrees from graph, only keep degrees in this partition
// std::vector is not a good option
template <typename T, typename GRAPH>
dense_state_t<T, typename GRAPH::partition_t> generate_dense_degrees_fg (
    const graph_info_t& graph_info,
    GRAPH&              graph,
    bool                is_out_degrees,
    bool                is_out_edge) {

  using partition_t          = typename GRAPH::partition_t;
  using adj_unit_spec_t      = typename GRAPH::adj_unit_spec_t;
  using adj_unit_list_spec_t = typename GRAPH::adj_unit_list_spec_t;

  std::vector<T> all_degrees(graph_info.max_v_i_ + 1, 0);

  auto e_traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
    if ((is_out_edge && is_out_degrees) || (false == is_out_edge && false == is_out_degrees)) {
      __sync_fetch_and_add(&all_degrees[v_i], adjs.end_ - adjs.begin_);
    } else {
      for (adj_unit_spec_t* it = adjs.begin_; it != adjs.end_; ++it) {
        __sync_fetch_and_add(&all_degrees[it->neighbour_], 1);
      }
    }
    return true;
  };

  graph.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 1;
    while (graph.next_chunk(e_traversal, &chunk_size)) { }
  }
  MPI_Allreduce(MPI_IN_PLACE, all_degrees.data(), graph_info.max_v_i_ + 1, get_mpi_data_type<T>(), MPI_SUM, MPI_COMM_WORLD);

  dense_state_t<T, partition_t> degrees(graph_info.max_v_i_, graph.partitioner());

  auto v_traversal = [&](vid_t v_i, vid_t* pval) {
    *pval = all_degrees[v_i];
    return true;
  };

  degrees.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 64;
    while (degrees.next_chunk(v_traversal, &chunk_size)) { }
  }

  return degrees;
}

template <typename T, typename GRAPH>
dense_state_t<T, typename GRAPH::partition_t> generate_dense_out_degrees_fg (
    const graph_info_t& graph_info,
    GRAPH&              graph,
    bool                is_out_edge) {
  return generate_dense_degrees_fg<T>(graph_info, graph, true, is_out_edge);
}

template <typename T, typename GRAPH>
dense_state_t<T, typename GRAPH::partition_t> generate_dense_in_degrees_fg (
    const graph_info_t& graph_info,
    GRAPH&              graph,
    bool                is_out_edge) {
  return generate_dense_degrees_fg<T>(graph_info, graph, false, is_out_edge);
}

/*
 * generate sparse out degrees
 *
 * \tparam T           integral type for count degrees
 * \tparam PART_IMPL   partitioner type
 * \tparam EDGE_CACHE  edges cache type
 *
 * \param graph_info   graph's info, usually generated from load_edges_cache
 * \param part         partitioner
 * \param cache        edges cache
 *
 * \return
 *    sparse state that hold vertex's degree belong to self partition
 **/
template <typename T, typename PART_IMPL, typename EDGE_CACHE>
sparse_state_t<T, PART_IMPL> generate_sparse_out_degrees (
    const graph_info_t& graph_info,
    std::shared_ptr<PART_IMPL> part,
    EDGE_CACHE& cache) {

  static_assert(std::is_integral<T>::value, "T can only be integral type");

  using edge_unit_spec_t = typename EDGE_CACHE::edge_unit_spec_t;
  struct degree_message_t {
    vid_t v_i_;
    T     degree_;
  };

  auto& cluster_info = cluster_info_t::get_instance();
  bitmap_t<> existed_v(graph_info.max_v_i_ + 1);
  std::unique_ptr<T[]> __degrees(new T[graph_info.max_v_i_ + 1]);
  memset(__degrees.get(), 0, sizeof(T) * (graph_info.max_v_i_ + 1));

  cache.reset_traversal();
  #pragma omp parallel
  {
    auto traversal = [&](size_t, edge_unit_spec_t* edge) {
      __sync_fetch_and_add(&__degrees[edge->src_], 1);
      existed_v.set_bit(edge->src_);
      existed_v.set_bit(edge->dst_);

      if (false == graph_info.is_directed_) {
        __sync_fetch_and_add(&__degrees[edge->dst_], 1);
      }
      return true;
    };

    size_t chunk_size = 64;
    while (cache.next_chunk(traversal, &chunk_size)) { }
  }

  std::atomic<size_t> degree_i(0);
  sparse_state_t<T, PART_IMPL> degrees((size_t)(graph_info.vertices_ / cluster_info.partitions_ * 2.0), part);

  auto __send = [&](bsp_send_callback_t<degree_message_t> send) {
    const size_t chunk_size = 64;

    size_t idx = 0;
    while ((idx = degree_i.fetch_add(chunk_size)) < (graph_info.max_v_i_ + 1)) {
      size_t end_i = idx + chunk_size;

      if (end_i > (graph_info.max_v_i_ + 1)) {
        end_i = graph_info.max_v_i_ + 1;
      }

      for (size_t i = idx; i < end_i; ++i) {
        if (existed_v.get_bit(i)) {
          send(part->get_partition_id(i), degree_message_t { (vid_t)i, __degrees[i] });
        }
      }
    }
  };

  auto __recv = [&](int, bsp_recv_pmsg_t<degree_message_t>& pmsg) {
    degrees.upsert(pmsg->v_i_, [&](T& degrees) { degrees += pmsg->degree_; }, pmsg->degree_);
  };

  bsp_opts_t opts;
  opts.global_size_    = 64 * MBYTES;
  opts.local_capacity_ = 32 * PAGESIZE;

  fine_grain_bsp<degree_message_t>(__send, __recv, opts);

  degrees.lock();
  return degrees;
}

// ******************************************************************************* //


// ******************************************************************************* //
// create a specific graph structure

/*
 * create dcsc graph structure with sequence balanced by source partition from file system
 *
 * \tparam EDATA          edge data type
 * \tparam SEQ_PART       sequence partition type
 * \tparam VID_T          vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE          cache type, can be edge_block_cache_t or edge_file_cache_t or edge_cache_t
 *
 * \param pgraph_info     user should fill 'is_directed_' field of  graph_info_t, this function
 *                        will fill other fields during load process.
 * \param path            input file path, 'path' can be a file or a directory.
 *                        'path' can be located on hdfs or posix, distinguish by its prefix.
 *                        eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format          file format
 * \param decoder         edge data decode, string => EDATA
 * \param is_directed     the graph is directed or not
 * \param alpha           vertex's weighted for partition, -1 means use default
 * \param use_in_degree   use in-degree instead of out degree for partition
 *
 * \return
 *      graph structure in dcsc form
 **/
template <typename EDATA, typename SEQ_PART, typename VID_T = vid_t,
         template<typename, typename> class CACHE = edge_block_cache_t>
std::shared_ptr<dcsc_t<EDATA, SEQ_PART>> create_dcsc_seq_from_path (
    graph_info_t*                       pgraph_info,
    const std::string&                  path,
    edge_format_t                       format,
    decoder_t<EDATA>                    decoder,
    int                                 alpha = -1,
    bool                                use_in_degree = false,
    vencoder_t<EDATA, VID_T, CACHE>     vid_encoder = nullptr) {

  static_assert(std::is_same<SEQ_PART, sequence_balanced_by_source_t>::value
      || std::is_same<SEQ_PART, sequence_balanced_by_destination_t>::value, "invalid SEQ_PART type!");

  using dcsc_spec_t = plato::dcsc_t<EDATA, SEQ_PART>;

  auto& cluster_info = cluster_info_t::get_instance();
  plato::stop_watch_t watch;

  watch.mark("t0");
  watch.mark("t1");

  auto cache = load_edges_cache<EDATA, VID_T, CACHE>(pgraph_info, path, format, decoder, nullptr, vid_encoder);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << pgraph_info->edges_;
    LOG(INFO) << "vertices:     " << pgraph_info->vertices_;
    LOG(INFO) << "max_v_id:     " << pgraph_info->max_v_i_;
    LOG(INFO) << "is_directed_: " << pgraph_info->is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::shared_ptr<SEQ_PART> part_dcsc = nullptr;
  {
    std::vector<vid_t> degrees;
    if (use_in_degree) {
      degrees = generate_dense_in_degrees<vid_t>(*pgraph_info, *cache);
    } else {
      degrees = generate_dense_out_degrees<vid_t>(*pgraph_info, *cache);
    }

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "generate degrees cost: " << watch.show("t1") / 1000.0 << "s";
    }
    watch.mark("t1");

    plato::eid_t __edges = pgraph_info->edges_;
    if (false == pgraph_info->is_directed_) { __edges = __edges * 2; }

    part_dcsc.reset(new SEQ_PART(degrees.data(), pgraph_info->vertices_,
      __edges, alpha));
    part_dcsc->check_consistency();
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "partition cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::shared_ptr<dcsc_spec_t> pdcsc(new dcsc_spec_t(part_dcsc));
  CHECK(0 == pdcsc->load_from_cache(*pgraph_info, *cache));

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build dcsc cost: " << watch.show("t1") / 1000.0 << "s";
    LOG(INFO) << "total cost:      " << watch.show("t0") / 1000.0 << "s";
  }

  return pdcsc;
}

// dcsc with sequence balanced partition by source
template <typename EDATA, typename VID_T = vid_t, template<typename, typename> class CACHE = edge_block_cache_t>
std::shared_ptr<dcsc_t<EDATA, sequence_balanced_by_source_t>> create_dcsc_seqs_from_path (
    graph_info_t*                   pgraph_info,
    const std::string&              path,
    edge_format_t                   format,
    decoder_t<EDATA>                decoder,
    int                             alpha = -1,
    bool                            use_in_degree = false,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr) {
  return create_dcsc_seq_from_path<EDATA, sequence_balanced_by_source_t, VID_T, CACHE>
    (pgraph_info, path, format, decoder, alpha, use_in_degree, vid_encoder);
}

// dcsc with sequence balanced partition by destination
template <typename EDATA, typename VID_T = vid_t, template<typename, typename> class CACHE = edge_block_cache_t>
std::shared_ptr<dcsc_t<EDATA, sequence_balanced_by_destination_t>> create_dcsc_seqd_from_path (
    graph_info_t*                   pgraph_info,
    const std::string&              path,
    edge_format_t                   format,
    decoder_t<EDATA>                decoder,
    int                             alpha = -1,
    bool                            use_in_degree = true,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr) {
  return create_dcsc_seq_from_path<EDATA, sequence_balanced_by_destination_t, VID_T, CACHE>
    (pgraph_info, path, format, decoder, alpha, use_in_degree, vid_encoder);
}

/*
 * create bcsr graph structure with sequence partition from file system
 *
 * \tparam EDATA          data type bind to edge
 * \tparam SEQ_PART       partitioner's type
 * \tparam VID_T          vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE          cache type, can be edge_block_cache_t or edge_file_cache_t or edge_cache_t
 *
 * \param pgraph_info     user should fill 'is_directed_' field of  graph_info_t, this function
 *                        will fill other fields during load process.
 * \param path            input file path, 'path' can be a file or a directory.
 *                        'path' can be located on hdfs or posix, distinguish by its prefix.
 *                        eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format          file format
 * \param decoder         edge data decode, string => EDATA
 * \param is_directed     the graph is directed or not
 * \param alpha           vertex's weighted for partition, -1 means use default
 * \param use_in_degree   use in-degree instead of out degree for partition
 * \param vid_encoder     embedded encoder for plato
 *
 * \return graph structure in bcsr form
 **/
template <typename EDATA, typename SEQ_PART, typename VID_T = vid_t,
         template<typename, typename> class CACHE = edge_block_cache_t>
std::shared_ptr<bcsr_t<EDATA, SEQ_PART>> create_bcsr_seq_from_path (
    graph_info_t*             pgraph_info,
    const std::string&        path,
    edge_format_t             format,
    decoder_t<EDATA>          decoder,
    int                       alpha = -1,
    bool                      use_in_degree = false,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr,
    bool                      is_outgoing = true) {

  static_assert(std::is_same<SEQ_PART, sequence_balanced_by_source_t>::value
      || std::is_same<SEQ_PART, sequence_balanced_by_destination_t>::value, "invalid SEQ_PART type!");

  using bcsr_spec_t = plato::bcsr_t<EDATA, SEQ_PART>;

  auto& cluster_info = cluster_info_t::get_instance();
  plato::stop_watch_t watch;

  watch.mark("t0");
  watch.mark("t1");

  auto cache = load_edges_cache<EDATA, VID_T, CACHE>(pgraph_info, path, format, decoder, nullptr, vid_encoder);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << pgraph_info->edges_;
    LOG(INFO) << "vertices:     " << pgraph_info->vertices_;
    LOG(INFO) << "max_v_id:     " << pgraph_info->max_v_i_;
    LOG(INFO) << "is_directed_: " << pgraph_info->is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::shared_ptr<SEQ_PART> part_bcsr = nullptr;
  {
    std::vector<vid_t> degrees;
    if (use_in_degree) {
      degrees = generate_dense_in_degrees<vid_t>(*pgraph_info, *cache);
    } else {
      degrees = generate_dense_out_degrees<vid_t>(*pgraph_info, *cache);
    }

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "generate degrees cost: " << watch.show("t1") / 1000.0 << "s";
    }
    watch.mark("t1");

    plato::eid_t __edges = pgraph_info->edges_;
    if (false == pgraph_info->is_directed_) { __edges = __edges * 2; }

    part_bcsr.reset(new SEQ_PART(degrees.data(), pgraph_info->vertices_,
      __edges, alpha));
    part_bcsr->check_consistency();
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "partition cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::shared_ptr<bcsr_spec_t> pbcsr(new bcsr_spec_t(part_bcsr));
  CHECK(0 == pbcsr->load_from_cache(*pgraph_info, *cache, is_outgoing));

  plato::mem_status_t mstatus;
  plato::self_mem_usage(&mstatus);

  LOG(INFO) << "memory usage(bcsr + cache): " << (double)mstatus.vm_rss / 1024.0 / 1024.0 << " GBytes";

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build bcsr cost: " << watch.show("t1") / 1000.0 << "s";
    LOG(INFO) << "total cost:      " << watch.show("t0") / 1000.0 << "s";
  }

  return pbcsr;
}

/*
 * create bcsr graph structure with sequence balanced by destination partition from file system
 *
 * \tparam EDATA          data type bind to edge
 * \tparam VID_T          vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE          cache type, can be edge_block_cache_t or edge_file_cache_t or edge_cache_t
 *
 * \param pgraph_info     user should fill 'is_directed_' field of  graph_info_t, this function
 *                        will fill other fields during load process.
 * \param path            input file path, 'path' can be a file or a directory.
 *                        'path' can be located on hdfs or posix, distinguish by its prefix.
 *                        eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format          file format
 * \param decoder         edge data decode, string => EDATA
 * \param is_directed     the graph is directed or not
 * \param alpha           vertex's weighted for partition, -1 means use default
 * \param use_in_degree   use in-degree instead of out degree for partition
 * \param vid_encoder     embedded encoder for plato
 *
 * \return graph structure in bcsr form
 **/
template <typename EDATA, typename VID_T = vid_t, template<typename, typename> class CACHE = edge_block_cache_t>
std::shared_ptr<bcsr_t<EDATA, sequence_balanced_by_destination_t>> create_bcsr_seqd_from_path (
    graph_info_t*             pgraph_info,
    const std::string&        path,
    edge_format_t             format,
    decoder_t<EDATA>          decoder,
    int                       alpha = -1,
    bool                      use_in_degree = false,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr,
    bool                      is_outgoing = true) {
  return create_bcsr_seq_from_path<EDATA, sequence_balanced_by_destination_t, VID_T, CACHE>
    (pgraph_info, path, format, decoder, alpha, use_in_degree, vid_encoder, is_outgoing);
}

/*
 * create bcsr graph structure with sequence balanced by source partition from file system
 *
 * \tparam EDATA          data type bind to edge
 * \tparam VID_T          vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE          cache type, can be edge_block_cache_t or edge_file_cache_t or edge_cache_t
 *
 * \param pgraph_info     user should fill 'is_directed_' field of  graph_info_t, this function
 *                        will fill other fields during load process.
 * \param path            input file path, 'path' can be a file or a directory.
 *                        'path' can be located on hdfs or posix, distinguish by its prefix.
 *                        eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format          file format
 * \param decoder         edge data decode, string => EDATA
 * \param is_directed     the graph is directed or not
 * \param alpha           vertex's weighted for partition, -1 means use default
 * \param use_in_degree   use in-degree instead of out degree for partition
 * \param vid_encoder     embedded encoder for plato
 *
 * \return graph structure in bcsr form
 **/
template <typename EDATA, typename VID_T = vid_t, template<typename, typename> class CACHE = edge_block_cache_t>
std::shared_ptr<bcsr_t<EDATA, sequence_balanced_by_source_t>> create_bcsr_seqs_from_path (
    graph_info_t*             pgraph_info,
    const std::string&        path,
    edge_format_t             format,
    decoder_t<EDATA>          decoder,
    int                       alpha = -1,
    bool                      use_in_degree = false,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr,
    bool                      is_outgoing = true) {
  return create_bcsr_seq_from_path<EDATA, sequence_balanced_by_source_t, VID_T, CACHE>
    (pgraph_info, path, format, decoder, alpha, use_in_degree, vid_encoder, is_outgoing);
}

/*
 * create tcsr graph structure with hash partition from file system
 *
 * \tparam EDATA          data type bind to edge
 * \tparam VDATA          data type bind to vertex
 * \tparam HASH_PART      hash-partitioner's type
 * \tparam VID_T          vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE          cache type, can be edge_block_cache_t or edge_file_cache_t or edge_cache_t
 *
 * \param pgraph_info     user should fill 'is_directed_' field of  graph_info_t, this function
 *                        will fill other fields during load process.
 * \param path            input file path, 'path' can be a file or a directory.
 *                        'path' can be located on hdfs or posix, distinguish by its prefix.
 *                        eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format          file format
 * \param decoder         edge data decode, string => EDATA
 *
 * \return graph structure in bcsr form
 **/
template <typename EDATA, typename VDATA, typename HASH_PART, typename VID_T = vid_t,
         template<typename, typename> class CACHE>
std::shared_ptr<tcsr_t<EDATA, VDATA, HASH_PART>> create_tcsr_hash_from_path (
  graph_info_t*                   pgraph_info,
  const std::string&              path,
  edge_format_t                   format,
  decoder_t<EDATA>                decoder,
  vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr) {

  auto& cluster_info = cluster_info_t::get_instance();
  HASH_PART partition;

  plato::stop_watch_t watch;
  watch.mark("t0");

  constexpr size_t mem_size = sizeof(vid_t) * (size_t(std::numeric_limits<vid_t>::max()) + 1);
  std::unique_ptr<vid_t, mmap_deleter> out_degree(
    (vid_t*)mmap(nullptr, mem_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
    mmap_deleter{mem_size});

  CHECK(MAP_FAILED != out_degree.get())
    << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno);

  watch.mark("t1");
  watch.mark("t2");

  auto callback = [&](edge_unit_t<EDATA, vid_t>* input, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      auto& edge = input[i];
      __sync_fetch_and_add(out_degree.get() + edge.src_, 1);
      if (!pgraph_info->is_directed_) {
        __sync_fetch_and_add(out_degree.get() + edge.dst_, 1);
      }
    }
    return true;
  };

  auto cache = load_edges_cache<EDATA, VID_T, CACHE>(pgraph_info, path, format, decoder, callback, vid_encoder);
  LOG(INFO) << "load edges cache cost: " << watch.show("t2") / 1000.0 << "s, partition: " << cluster_info.partition_id_;


  auto max_vid = pgraph_info->max_v_i_;
  {
    // count degree
    // reduce communication network compared with MPI_Allreduce
    watch.mark("t2");
    bsp_opts_t opts;
    opts.threads_               = -1;
    opts.flying_send_per_node_  = 2;
    opts.flying_recv_           = cluster_info.threads_;
    opts.global_size_           = 16 * MBYTES;
    opts.local_capacity_        = 32 * PAGESIZE;
    opts.batch_size_            = 1;

    struct degree_msg {
      plato::vid_t dst_;
      plato::vid_t degree_;
    };

    size_t vid = 0;
    auto __send = [&] (bsp_send_callback_t<degree_msg> send) {
      while (true) {
        size_t begin = __sync_fetch_and_add(&vid, MBYTES);
        if (begin > max_vid) break;
        size_t end = std::min(size_t(max_vid) + 1, begin + MBYTES);
        for (size_t v_i = begin; v_i < end; v_i++) {
          vid_t degree = *(out_degree.get() + v_i);
          if (degree) {
            int partition_id = partition.get_partition_id(v_i);
            if (partition_id != cluster_info.partition_id_) {
              degree_msg msg;
              msg.dst_ = v_i;
              msg.degree_ = degree;
              send(partition_id, msg);
            }
          }
        }
      }
    };

    auto __recv = [&] (int /*p_i*/, bsp_recv_pmsg_t<degree_msg>& pmsg) {
      degree_msg& msg = *pmsg;
      CHECK(partition.get_partition_id(msg.dst_) == cluster_info.partition_id_);
      __sync_fetch_and_add(out_degree.get() + msg.dst_, msg.degree_);
    };

    auto rc = fine_grain_bsp<degree_msg>(__send, __recv, opts);
    if (0 != rc) {
      LOG(ERROR) << "bsp failed with code: " << rc;
      return nullptr;
    }
    LOG_IF(INFO, 0 == cluster_info.partition_id_) << "reduce out_degree cost: " << watch.show("t2") / 1000.0 << "s";
  }

  watch.mark("t2");
  size_t local_degree_sum = 0;
  size_t local_vertices = 0;
  #pragma omp parallel for reduction(+:local_degree_sum) reduction(+:local_vertices)
  for (size_t v_i = 0; v_i <= max_vid; v_i++) {
    if (partition.get_partition_id(v_i) == cluster_info.partition_id_ && *(out_degree.get() + v_i)) {
      local_degree_sum += *(out_degree.get() + v_i);
      local_vertices++;
    }
  }
  size_t degree_sum = 0;
  size_t vertices = 0;
  MPI_Allreduce(&local_degree_sum, &degree_sum, 1, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_vertices, &vertices, 1, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
  CHECK(pgraph_info->is_directed_ ? degree_sum == pgraph_info->edges_ : degree_sum == pgraph_info->edges_ * 2);

  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "count degree sum & vertices cost: " << watch.show("t2") / 1000.0 << "s";
  LOG(INFO) << "[partition-" << cluster_info.partition_id_ << "] local_degree_sum: " << local_degree_sum
    << ", local_vertices: " << local_vertices;

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << pgraph_info->edges_;
    LOG(INFO) << "vertices:     " << pgraph_info->vertices_;
    LOG(INFO) << "max_v_id:     " << pgraph_info->max_v_i_;
    LOG(INFO) << "is_directed_: " << pgraph_info->is_directed_;
    LOG(INFO) << "degree_sum:   " << degree_sum;

    LOG(INFO) << "load edges cache & bitmap & out_degree cost: " << watch.show("t1") / 1000.0 << "s";
  }

  watch.mark("t1");

  using tcsr_spec_t = plato::tcsr_t<EDATA, VDATA, HASH_PART>;
  std::shared_ptr<tcsr_spec_t> ptcsr(new tcsr_spec_t(local_vertices * 1.2, std::shared_ptr<HASH_PART>(new HASH_PART())));

  CHECK(0 == ptcsr->load_edges_from_cache(*pgraph_info, *cache, std::move(out_degree), local_degree_sum));

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build tcsr cost: " << watch.show("t1") / 1000.0 << "s";
    LOG(INFO) << "total cost:      " << watch.show("t0") / 1000.0 << "s";
  }

  return ptcsr;
}

/*
 * create tcsr graph structure with hash balanced by destination partition from file system
 *
 * \tparam EDATA          data type bind to edge
 * \tparam VDATA          data type bind to vertex
 * \tparam Hash           hash function used by vertex partitioner
 * \tparam VID_T          vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE          cache type, can be edge_cache_t or edge_file_cache_t
 *
 * \param pgraph_info     user should fill 'is_directed_' field of  graph_info_t, this function
 *                        will fill other fields during load process.
 * \param path            input file path, 'path' can be a file or a directory.
 *                        'path' can be located on hdfs or posix, distinguish by its prefix.
 *                        eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format          file format
 * \param decoder         edge data decode, string => EDATA
 *
 * \return graph structure in bcsr form
 **/
template <typename EDATA, typename VDATA, typename Hash = std::hash<vid_t>, typename VID_T = vid_t,
         template<typename, typename> class CACHE = edge_file_cache_t>
std::shared_ptr<tcsr_t<EDATA, VDATA, hash_by_destination_t<Hash>>> create_tcsr_hashd_from_path (
    graph_info_t*                   pgraph_info,
    const std::string&              path,
    edge_format_t                   format,
    decoder_t<EDATA>                decoder,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr) {
  return create_tcsr_hash_from_path<EDATA, VDATA, hash_by_destination_t<Hash>, VID_T, CACHE>
    (pgraph_info, path, format, decoder, vid_encoder);
}

/*
 * create tcsr graph structure with hash balanced by source partition from file system
 *
 * \tparam EDATA          data type bind to edge
 * \tparam VDATA          data type bind to vertex
 * \tparam Hash           hash function used by vertex partitioner
 * \tparam VID_T          vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE          cache type, can be edge_block_cache_t or edge_file_cache_t or edge_cache_t
 *
 * \param pgraph_info     user should fill 'is_directed_' field of  graph_info_t, this function
 *                        will fill other fields during load process.
 * \param path            input file path, 'path' can be a file or a directory.
 *                        'path' can be located on hdfs or posix, distinguish by its prefix.
 *                        eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format          file format
 * \param decoder         edge data decode, string => EDATA
 *
 * \return graph structure in tcsr form
 **/
template <typename EDATA, typename VDATA, typename Hash = std::hash<vid_t>, typename VID_T = vid_t,
         template<typename, typename> class CACHE = edge_file_cache_t>
std::shared_ptr<tcsr_t<EDATA, VDATA, hash_by_source_t<Hash>>> create_tcsr_hashs_from_path (
    graph_info_t*                   pgraph_info,
    const std::string&              path,
    edge_format_t                   format,
    decoder_t<EDATA>                decoder,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr) {
  return create_tcsr_hash_from_path<EDATA, VDATA, hash_by_source_t<Hash>, VID_T, CACHE>
    (pgraph_info, path, format, decoder, vid_encoder);
}

/*
 * create bcsr and dcsc graph structure from file system
 * bcsr's is partition by sequence balanced by destination
 * dcsc's is partition by sequence balanced by source
 *
 * \tparam EDATA          data type bind to edge
 * \tparam VID_T          vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE          cache type, can be edge_block_cache_t or edge_file_cache_t or edge_cache_t
 *
 * \param pgraph_info     user should fill 'is_directed_' field of  graph_info_t, this function
 *                        will fill other fields during load process.
 * \param path            input file path, 'path' can be a file or a directory.
 *                        'path' can be located on hdfs or posix, distinguish by its prefix.
 *                        eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format          file format
 * \param decoder         edge data decode, string => EDATA
 * \param is_directed     the graph is directed or not
 * \param alpha           vertex's weighted for partition, -1 means use default
 * \param use_in_degree   use in-degree instead of out degree for partition
 *
 * \return graph structure in dual-mode, first -- bcsr, second -- dcsc
 **/
template <typename EDATA, typename VID_T = vid_t, template<typename, typename> class CACHE = edge_block_cache_t>
std::pair<
  bcsr_t<EDATA, sequence_balanced_by_destination_t>,
  dcsc_t<EDATA, sequence_balanced_by_source_t>
> create_dualmode_seq_from_path (
    graph_info_t*                   pgraph_info,
    const std::string&              path,
    edge_format_t                   format,
    decoder_t<EDATA>                decoder,
    int                             alpha = -1,
    bool                            use_in_degree = false,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr) {

  using bcsr_spec_t = plato::bcsr_t<EDATA, plato::sequence_balanced_by_destination_t>;
  using dcsc_spec_t = plato::dcsc_t<EDATA, plato::sequence_balanced_by_source_t>;

  auto& cluster_info = cluster_info_t::get_instance();
  plato::stop_watch_t watch;

  watch.mark("t0");
  watch.mark("t1");

  auto cache = load_edges_cache<EDATA, VID_T, CACHE>(pgraph_info, path, format, decoder, nullptr, vid_encoder);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "path:         " << path;
    LOG(INFO) << "edges:        " << pgraph_info->edges_;
    LOG(INFO) << "vertices:     " << pgraph_info->vertices_;
    LOG(INFO) << "max_v_id:     " << pgraph_info->max_v_i_;
    LOG(INFO) << "is_directed_: " << pgraph_info->is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::shared_ptr<plato::sequence_balanced_by_destination_t> part_bcsr = nullptr;
  {
    std::vector<vid_t> degrees;
    if (use_in_degree) {
      degrees = generate_dense_in_degrees<vid_t>(*pgraph_info, *cache);
    } else {
      degrees = generate_dense_out_degrees<vid_t>(*pgraph_info, *cache);
    }

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "generate degrees cost: " << watch.show("t1") / 1000.0 << "s";
    }
    watch.mark("t1");

    plato::eid_t __edges = pgraph_info->edges_;
    if (false == pgraph_info->is_directed_) { __edges = __edges * 2; }

    part_bcsr.reset(
        new plato::sequence_balanced_by_destination_t(degrees.data(), pgraph_info->vertices_,
          __edges, alpha)
    );
    part_bcsr->check_consistency();
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "partition cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  bcsr_spec_t bcsr(part_bcsr);
  CHECK(0 == bcsr.load_from_cache(*pgraph_info, *cache));

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build bcsr cost: " << watch.show("t1") / 1000.0 << "s";
  }

  plato::mem_status_t mstatus;
  plato::self_mem_usage(&mstatus);

  LOG(INFO) << "memory usage(bcsr + cache): " << (double)mstatus.vm_rss / 1024.0 / 1024.0 << " GBytes";

  cache = nullptr;  // destroy edges cache

  plato::self_mem_usage(&mstatus);
  LOG(INFO) << "memory usage(bcsr): " << (double)mstatus.vm_rss / 1024.0 / 1024.0 << " GBytes";

  std::shared_ptr<plato::sequence_balanced_by_source_t> part_dcsc (
      new plato::sequence_balanced_by_source_t(part_bcsr->offset_)
  );

  watch.mark("t1");

  dcsc_spec_t dcsc(part_dcsc);

  {
    plato::traverse_opts_t opts; opts.mode_ = plato::traverse_mode_t::RANDOM;
    CHECK(0 == dcsc.load_from_graph(*pgraph_info, bcsr, true, opts));
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build dcsc cost:     " << watch.show("t1") / 1000.0 << "s";
    LOG(INFO) << "build dualmode cost: " << watch.show("t0") / 1000.0 << "s";
  }

  plato::self_mem_usage(&mstatus);
  LOG(INFO) << "memory usage(bcsr + dcsc): " << (double)mstatus.vm_rss / 1024.0 / 1024.0 << " GBytes";

  return std::make_pair(std::move(bcsr), std::move(dcsc));
}

/*
 * create bcsr graph structure from file system
 * outgoing is partition by sequence balanced by destination
 * incoming is partition by sequence balanced by source
 *
 * \tparam EDATA          data type bind to edge
 * \tparam VID_T          vertex id type, can be uint32_t or uint64_t
 * \tparam CACHE          cache type, can be edge_block_cache_t or edge_file_cache_t or edge_cache_t
 *
 * \param pgraph_info     user should fill 'is_directed_' field of  graph_info_t, this function
 *                        will fill other fields during load process.
 * \param path            input file path, 'path' can be a file or a directory.
 *                        'path' can be located on hdfs or posix, distinguish by its prefix.
 *                        eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format          file format
 * \param decoder         edge data decode, string => EDATA
 * \param is_directed     the graph is directed or not
 * \param alpha           vertex's weighted for partition, -1 means use default
 * \param use_in_degree   use in-degree instead of out degree for partition
 *
 * \return graph structure in dual-mode, first -- outgoing, second -- incoming
 **/
template <typename EDATA, typename VID_T = vid_t, template<typename, typename> class CACHE = edge_block_cache_t>
std::pair<
  bcsr_t<EDATA, sequence_balanced_by_destination_t>,
  bcsr_t<EDATA, sequence_balanced_by_destination_t>
> create_dualmode_bcsr_seq_from_path (
    graph_info_t*                   pgraph_info,
    const std::string&              path,
    edge_format_t                   format,
    decoder_t<EDATA>                decoder,
    int                             alpha = -1,
    bool                            use_in_degree = false,
    vencoder_t<EDATA, VID_T, CACHE> vid_encoder = nullptr) {

  using outgoing_t = plato::bcsr_t<EDATA, plato::sequence_balanced_by_destination_t>;
  using incoming_t = plato::bcsr_t<EDATA, plato::sequence_balanced_by_destination_t>;

  auto& cluster_info = cluster_info_t::get_instance();
  plato::stop_watch_t watch;

  watch.mark("t0");
  watch.mark("t1");

  auto cache = load_edges_cache<EDATA, VID_T, CACHE>(pgraph_info, path, format, decoder, nullptr, vid_encoder);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "path:         " << path;
    LOG(INFO) << "edges:        " << pgraph_info->edges_;
    LOG(INFO) << "vertices:     " << pgraph_info->vertices_;
    LOG(INFO) << "max_v_id:     " << pgraph_info->max_v_i_;
    LOG(INFO) << "is_directed_: " << pgraph_info->is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::vector<vid_t> degrees;
  if (use_in_degree) {
    degrees = generate_dense_in_degrees<vid_t>(*pgraph_info, *cache);
  } else {
    degrees = generate_dense_out_degrees<vid_t>(*pgraph_info, *cache);
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "generate degrees cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  plato::eid_t __edges = pgraph_info->edges_;
  if (false == pgraph_info->is_directed_) { __edges = __edges * 2; }

  std::shared_ptr<plato::sequence_balanced_by_destination_t> part_out (
      new plato::sequence_balanced_by_destination_t(degrees.data(), pgraph_info->vertices_,
        __edges, alpha)
  );
  part_out->check_consistency();

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "partition cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  outgoing_t outgoing(part_out);
  CHECK(0 == outgoing.load_from_cache(*pgraph_info, *cache));

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build outgoing cost: " << watch.show("t1") / 1000.0 << "s";
  }

  plato::mem_status_t mstatus;
  plato::self_mem_usage(&mstatus);

  LOG(INFO) << "memory usage(outgoing + cache): " << (double)mstatus.vm_rss / 1024.0 / 1024.0 << " GBytes";

  cache = nullptr;  // destroy edges cache

  plato::self_mem_usage(&mstatus);
  LOG(INFO) << "memory usage(outgoing): " << (double)mstatus.vm_rss / 1024.0 / 1024.0 << " GBytes";

  std::shared_ptr<plato::sequence_balanced_by_destination_t> part_in (
      new plato::sequence_balanced_by_destination_t(part_out->offset_)
  );

  watch.mark("t1");

  incoming_t incoming(part_in);

  {
    plato::traverse_opts_t opts; opts.mode_ = plato::traverse_mode_t::RANDOM;

    // we fake incoming here to make bcsr load incoming from bcsr
    CHECK(0 == incoming.load_from_graph(*pgraph_info, outgoing, false, opts));
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build incoming cost: " << watch.show("t1") / 1000.0 << "s";
    LOG(INFO) << "build dualmode cost: " << watch.show("t0") / 1000.0 << "s";
  }

  plato::self_mem_usage(&mstatus);
  LOG(INFO) << "memory usage(outgoing + incoming): " << (double)mstatus.vm_rss / 1024.0 / 1024.0 << " GBytes";

  return std::make_pair(std::move(outgoing), std::move(incoming));
}

// ******************************************************************************* //

}  // namespace plato

#endif

