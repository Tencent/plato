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

#include "boost/format.hpp"
#include "sparsehash/dense_hash_map"

#include "plato/util/perf.hpp"
#include "plato/util/thread_local_object.h"
#include "plato/util/object_buffer.hpp"
#include "plato/graph/base.hpp"
#include "plato/graph/structure.hpp"
#include "plato/graph/state/sparse_state.hpp"

DEFINE_string(input_relation_edges,      "",         "input relation edges file, in csv format");
DEFINE_string(input_behaviour_edges,     "",         "input behaviour edges file, in csv format");
DEFINE_string(output,                    "",         "output directory");
DEFINE_bool(output_list,                 false,      "true: output list, false: output count.");
DEFINE_bool(is_directed,                 false,      "relation is directed.");
DEFINE_uint64(split_factor,              3,          "split ${FLAGS_split_factor} groups to load graph, in order to use less memory.");
DEFINE_string(cache_dir,                 ".cache",   "use path start with hdfs:// to store cache in hdfs, otherwise use posix file storage instead.");
DEFINE_uint64(sum_items_num,             0,          "how many items need to sum.");

bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

DEFINE_validator(input_relation_edges,  &string_not_empty);
DEFINE_validator(input_behaviour_edges,  &string_not_empty);
DEFINE_validator(output,  &string_not_empty);
DEFINE_validator(cache_dir,  &string_not_empty);
DEFINE_validator(sum_items_num,  [] (const char*, uint64_t value) { return value <= 5; });

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

using edge_t = plato::empty_t;
std::unique_ptr<plato::thread_local_buffer> g_adjs_buffer_p_;

struct relation_t {
  plato::vid_t uin_;
  plato::vid_t adjs_size_;
  plato::vid_t* adjs_ = nullptr;

  template<typename Ar>
  void serialize(Ar &ar) {
    if(!adjs_) adjs_ = (plato::vid_t*)g_adjs_buffer_p_->local();
    ar & uin_;
    ar & adjs_size_;
    for (unsigned i = 0; i < adjs_size_; i++) {
      ar & adjs_[i];
    }
  }
};

using partition_t = plato::hash_by_source_t<plato::cuckoo_vid_hash>;

struct group_partition_t {
  unsigned groups_;
  unsigned group_id_;
  partition_t partition;

  group_partition_t(unsigned groups, unsigned group_id) :
    groups_(groups), group_id_(group_id) {};

  int get_partition_id(plato::vid_t v, plato::vid_t = 0) {
    plato::vid_t v_i = v * groups_ + group_id_;
    return partition.get_partition_id(v_i);
  }
};

struct degree_unit {
  plato::vid_t vid_;
  plato::vid_t degree_;
};

template <template <typename, typename Enable = void> class OBJECT_BUFFER>
OBJECT_BUFFER<relation_t> load_realation_graph(plato::graph_info_t& relations_graph_info, plato::bitmap_t<>&& behaviour_bitmap) {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  partition_t partition;

  plato::stop_watch_t watch;
  watch.mark("t0");
  watch.mark("t1");

  std::vector<std::unique_ptr<plato::vid_t, plato::mmap_deleter>> out_degree_v(FLAGS_split_factor);
  size_t out_degree_max_num_per_group = (size_t(std::numeric_limits<plato::vid_t>::max()) + 1) / FLAGS_split_factor + 1;
  {
    size_t out_degree_max_mem_size_per_group = sizeof(plato::vid_t) * out_degree_max_num_per_group;
    for (auto& out_degree : out_degree_v) {
      out_degree = std::unique_ptr<plato::vid_t, plato::mmap_deleter>(
        (plato::vid_t*)mmap(nullptr, out_degree_max_mem_size_per_group, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
        plato::mmap_deleter{out_degree_max_mem_size_per_group});
      CHECK(MAP_FAILED != out_degree.get())
      << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno);
    }
  }

  plato::object_buffer_opt_t relations_buffer_opt;
  relations_buffer_opt.path_ = FLAGS_cache_dir;
  relations_buffer_opt.prefix_ = (boost::format("relation-graph-%04ld-") % cluster_info.partition_id_).str();
  relations_buffer_opt.capacity_ = (size_t)sysconf(_SC_PAGESIZE) * (size_t)sysconf(_SC_PHYS_PAGES) * FLAGS_split_factor * 1.5;

  OBJECT_BUFFER<relation_t> relations(relations_buffer_opt);
  std::vector<OBJECT_BUFFER<plato::edge_unit_t<edge_t>>> cache_v;
  std::vector<OBJECT_BUFFER<degree_unit>> out_degree_cache_v;

  {
    cache_v.reserve(FLAGS_split_factor);
    out_degree_cache_v.reserve(FLAGS_split_factor);
    for (unsigned group = 0; group < FLAGS_split_factor; ++group) {
      plato::object_buffer_opt_t relations_sub_edge_buffer_opt;
      relations_sub_edge_buffer_opt.path_ = FLAGS_cache_dir;
      relations_sub_edge_buffer_opt.prefix_ = (boost::format("relation-sub-%04lu-edge-%04ld-") % group % cluster_info.partition_id_).str();
      relations_sub_edge_buffer_opt.capacity_ = (size_t)sysconf(_SC_PAGESIZE) * (size_t)sysconf(_SC_PHYS_PAGES) * 2 / sizeof(plato::edge_unit_t<edge_t>) * 1.5;

      plato::object_buffer_opt_t relations_sub_degree_buffer_opt;
      relations_sub_degree_buffer_opt.path_ = FLAGS_cache_dir;
      relations_sub_degree_buffer_opt.prefix_ = (boost::format("relation-sub-%04lu-degree-%04ld-") % group % cluster_info.partition_id_).str();
      relations_sub_degree_buffer_opt.capacity_ = out_degree_max_num_per_group;

      cache_v.emplace_back(relations_sub_edge_buffer_opt);
      out_degree_cache_v.emplace_back(relations_sub_degree_buffer_opt);
    }
  }

  std::mutex mutex;
  std::vector<std::string> files = plato::get_files(FLAGS_input_relation_edges);
  plato::vid_t max_vid = 0;
  plato::eid_t edges = 0;
  plato::eid_t valid_directed_edges = 0;
  plato::vid_t valid_max_vid = 0;

  {
    watch.mark("t2");
    #pragma omp parallel reduction(max:max_vid) reduction(max:valid_max_vid) reduction(+:edges) reduction(+:valid_directed_edges)
    {
      while (true) {
        std::string filename;
        {
          std::lock_guard<std::mutex> lock(mutex);
          if (files.empty()) break;
          filename = std::move(files.back());
          files.pop_back();
        }

        plato::with_file(filename, [&] (boost::iostreams::filtering_istream& is) {
          plato::csv_parser<boost::iostreams::filtering_istream, edge_t, plato::vid_t>(
            is,
            [&] (plato::edge_unit_t<edge_t >* input, size_t size) {
              edges += size;

              for (size_t i = 0; i < size; ++i) {
                auto& edge = input[i];

                {
                  if (behaviour_bitmap.get_bit(edge.dst_)) {
                    unsigned group = edge.src_ % FLAGS_split_factor;
                    unsigned group_offset = edge.src_ / FLAGS_split_factor;
                    __sync_fetch_and_add(out_degree_v[group].get() + group_offset, 1);
                    plato::edge_unit_t<edge_t> e{group_offset, {edge.dst_}};
                    cache_v[group].push_back(e);
                    valid_directed_edges++;
                    valid_max_vid = std::max(std::max(edge.src_, edge.dst_), valid_max_vid);
                  }
                }

                if (!relations_graph_info.is_directed_) {
                  std::swap(edge.src_, edge.dst_);

                  if (behaviour_bitmap.get_bit(edge.dst_)) {
                    unsigned group = edge.src_ % FLAGS_split_factor;
                    unsigned group_offset = edge.src_ / FLAGS_split_factor;
                    __sync_fetch_and_add(out_degree_v[group].get() + group_offset, 1);
                    plato::edge_unit_t<edge_t> e{group_offset, {edge.dst_}};
                    cache_v[group].push_back(e);
                    valid_directed_edges++;
                    valid_max_vid = std::max(std::max(edge.src_, edge.dst_), valid_max_vid);
                  }
                }

                max_vid = std::max(std::max(edge.src_, edge.dst_), max_vid);
              }
              return true;
            },
            plato::dummy_decoder<edge_t>
          );
        });
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &edges, 1, plato::get_mpi_data_type<plato::eid_t>(), MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &valid_directed_edges, 1, plato::get_mpi_data_type<plato::eid_t>(), MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_vid, 1, plato::get_mpi_data_type<plato::vid_t>(), MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &valid_max_vid, 1, plato::get_mpi_data_type<plato::vid_t>(), MPI_MAX, MPI_COMM_WORLD);
    LOG(INFO) << "load realation cache cost: " << watch.show("t2") / 1000.0 << "s";
  }

  std::vector<size_t> local_degree_sum_v(FLAGS_split_factor, 0);
  std::vector<size_t> local_vertices_v(FLAGS_split_factor, 0);
  std::vector<size_t> local_max_vid_v(FLAGS_split_factor, 0);
  std::vector<size_t> degree_sum_v(FLAGS_split_factor, 0);
  std::vector<size_t> vertices_v(FLAGS_split_factor, 0);
  std::vector<size_t> max_vid_v(FLAGS_split_factor, 0);
  size_t degree_sum = 0, vertices = 0;

  {
    watch.mark("t2");

    size_t vid = 0;
    auto __send = [&] (plato::bsp_send_callback_t<degree_unit> send) {
      while (true) {
        size_t begin = __sync_fetch_and_add(&vid, MBYTES);
        if (begin > max_vid) break;
        size_t end = std::min(size_t(max_vid) + 1, begin + MBYTES);
        for (size_t v_i = begin; v_i < end; v_i++) {
          unsigned group = v_i % FLAGS_split_factor;
          unsigned group_offset = v_i / FLAGS_split_factor;

          plato::vid_t degree = *(out_degree_v[group].get() + group_offset);
          if (degree) {
            int partition_id = partition.get_partition_id(v_i);
            if (partition_id != cluster_info.partition_id_) {
              degree_unit v{plato::vid_t(v_i), degree};
              send(partition_id, v);
            }
          }
        }
      }
    };

    auto __recv = [&] (int /*p_i*/, plato::bsp_recv_pmsg_t<degree_unit>& pv) {
      degree_unit& v = *pv;
      CHECK(partition.get_partition_id(v.vid_) == cluster_info.partition_id_);
      unsigned group = v.vid_ % FLAGS_split_factor;
      unsigned group_offset = v.vid_ / FLAGS_split_factor;
      __sync_fetch_and_add(out_degree_v[group].get() + group_offset, v.degree_);
    };

    auto rc = plato::fine_grain_bsp<degree_unit>(__send, __recv);
    CHECK(0 == rc) << "bsp failed with code: " << rc;

    std::vector<plato::thread_local_counter> local_degree_sum_counter_v(FLAGS_split_factor);
    std::vector<plato::thread_local_counter> local_vertices_counter_v(FLAGS_split_factor);
    std::vector<plato::thread_local_counter> local_max_vid_counter_v(FLAGS_split_factor);

    #pragma omp parallel for
    for (size_t v_i = 0; v_i <= max_vid; v_i++) {
      if (partition.get_partition_id(v_i) == cluster_info.partition_id_) {
        unsigned group = v_i % FLAGS_split_factor;
        unsigned group_offset = v_i / FLAGS_split_factor;
        auto degree = *(out_degree_v[group].get() + group_offset);
        if (degree) {
          local_degree_sum_counter_v[group].local() += degree;
          local_vertices_counter_v[group].local()++;
          local_max_vid_counter_v[group].local() = v_i;
          degree_unit v{plato::vid_t(v_i), degree};
          out_degree_cache_v[group].push_back(v);
        }
      }
    }
    out_degree_v.clear();

    for (unsigned group = 0; group < FLAGS_split_factor; ++group) {
      local_degree_sum_v[group] = local_degree_sum_counter_v[group].reduce_sum();
      local_vertices_v[group] = local_vertices_counter_v[group].reduce_sum();
      local_max_vid_v[group] = local_max_vid_counter_v[group].reduce_max();
    }

    MPI_Allreduce(local_degree_sum_v.data(), degree_sum_v.data(), FLAGS_split_factor, plato::get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_vertices_v.data(), vertices_v.data(), FLAGS_split_factor, plato::get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_max_vid_v.data(), max_vid_v.data(), FLAGS_split_factor, plato::get_mpi_data_type<size_t>(), MPI_MAX, MPI_COMM_WORLD);

    degree_sum = std::accumulate(degree_sum_v.begin(), degree_sum_v.end(), 0UL);
    vertices = std::accumulate(vertices_v.begin(), vertices_v.end(), 0UL);
    CHECK(degree_sum == valid_directed_edges);
    LOG_IF(INFO, 0 == cluster_info.partition_id_) << "reduce realation out_degree cost: " << watch.show("t2") / 1000.0 << "s";
  }

  relations_graph_info.edges_    = valid_directed_edges;
  relations_graph_info.vertices_ = vertices;
  relations_graph_info.max_v_i_  = valid_max_vid;

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:          " << edges;
    LOG(INFO) << "valid edges:    " << relations_graph_info.edges_;
    LOG(INFO) << "vertices:       " << relations_graph_info.vertices_;
    LOG(INFO) << "max_v_id:       " << max_vid;
    LOG(INFO) << "valid max_v_id: " << relations_graph_info.max_v_i_;
    LOG(INFO) << "is_directed_:   " << relations_graph_info.is_directed_;
    LOG(INFO) << "degree_sum:     " << degree_sum;

    LOG(INFO) << "load realation cache & count out_degree cost: " << watch.show("t1") / 1000.0 << "s";
  }

  watch.mark("t1");

  using tcsr_spec_t = plato::tcsr_t<edge_t, plato::empty_t, group_partition_t>;
  for (unsigned group = 0; group < FLAGS_split_factor; group++) {
    watch.mark("t2");
    watch.mark("t3");

    size_t group_local_vertices = local_vertices_v[group];
    size_t group_local_degree_sum = local_degree_sum_v[group];
    size_t group_degree_sum = degree_sum_v[group];
    size_t group_max_vid = max_vid_v[group];

    LOG(INFO) << boost::format("group: %lu, partition: %d, local_vertices: %lu, degree_sum: %lu") % group % cluster_info.partition_id_ % group_local_vertices % group_local_degree_sum;
    tcsr_spec_t tcsr(group_local_vertices * 1.2, std::make_shared<group_partition_t>(FLAGS_split_factor, group));

    plato::graph_info_t g_info;
    g_info.edges_    = group_degree_sum;
    g_info.vertices_ = group_local_vertices;
    g_info.max_v_i_  = group_max_vid / FLAGS_split_factor;
    g_info.is_directed_ = true;

    std::unique_ptr<plato::vid_t, plato::mmap_deleter> out_degree(
      (plato::vid_t*)mmap(nullptr, (g_info.max_v_i_ + 1) * sizeof(plato::vid_t), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
      plato::mmap_deleter{(g_info.max_v_i_ + 1) * sizeof(plato::vid_t)});
    CHECK(MAP_FAILED != out_degree.get())
    << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno);

    {
      size_t group_local_degree_sum_ = 0;
      size_t group_local_vertices_ = 0;
      out_degree_cache_v[group].reset_traversal();
      #pragma omp parallel reduction(+:group_local_degree_sum_) reduction(+:group_local_vertices_)
      {
        size_t chunk_size = 1;
        while (out_degree_cache_v[group].next_chunk([&] (size_t, degree_unit* v) {
          out_degree.get()[v->vid_ / FLAGS_split_factor] = v->degree_;
          group_local_degree_sum_ += v->degree_;
          group_local_vertices_++;
        }, &chunk_size)) {}
      }

      CHECK(group_local_degree_sum_ == group_local_degree_sum)
      << boost::format("group_local_degree_sum_: %lu, group_local_degree_sum: %lu") % group_local_degree_sum_ % group_local_degree_sum;
      CHECK(group_local_vertices_ == group_local_vertices);
    }

    CHECK(0 == tcsr.load_edges_from_cache(g_info, cache_v[group], std::move(out_degree), group_local_degree_sum));
    LOG_IF(INFO, 0 == cluster_info.partition_id_) << boost::format("group: %lu, load tcsr cost: %.4fs") % group % (watch.show("t3") / 1000.0);

    watch.mark("t3");

    tcsr.reset_traversal();
    size_t chunk_size = 64;
    while (tcsr.next_chunk([&] (plato::vid_t group_offset, const plato::adj_unit_list_t<edge_t>& adjs) {
      relation_t relation;
      relation.uin_ = group_offset * FLAGS_split_factor + group;
      relation.adjs_size_ = adjs.end_ - adjs.begin_;
      relation.adjs_ = (plato::vid_t*)adjs.begin_;
      relations.push_back(relation);
    }, &chunk_size)) { }

    LOG_IF(INFO, 0 == cluster_info.partition_id_) << boost::format("group: %lu, save tcsr to file cost: %.4lfs") % group % (watch.show("t3") / 1000.0);
    LOG_IF(INFO, 0 == cluster_info.partition_id_) << boost::format("group: %lu, build object file csr cost: %.4lfs") % group % (watch.show("t2") / 1000.0);
  }

  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "load object file csr cost: " << watch.show("t1") / 1000.0 << "s";
  return relations;
}

template <unsigned sum_items_num>
struct behaviour_state_content_t {
  uint64_t behaviour_id_;
  uint32_t sum_items_[sum_items_num];
};

template <unsigned sum_items_num>
struct behaviour_state_t {
  plato::vid_t behaviour_size_;
  behaviour_state_content_t<sum_items_num>* behaviour_;

  behaviour_state_t(const behaviour_state_t&) = delete;
  behaviour_state_t& operator=(const behaviour_state_t&) = delete;
  behaviour_state_t(behaviour_state_t&& x) = default;
  behaviour_state_t& operator=(behaviour_state_t&& x) = default;

  behaviour_state_t(behaviour_state_content_t<sum_items_num>* behaviour) noexcept : behaviour_size_(0), behaviour_(behaviour) { }
};

static std::shared_ptr<void> behaviour_mmap_p;

int get_behaviour_partition_id(uint64_t id) {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  return ((uint64_t)id) % cluster_info.partitions_;
}

template <unsigned sum_items_num, template <typename, typename Enable = void> class OBJECT_BUFFER>
void load_behaviour_graph_cache(
  OBJECT_BUFFER<plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>>& behaviour_cache,
  plato::graph_info_t& behaviour_graph_info,
  OBJECT_BUFFER<degree_unit>& behaviour_degree_cache, plato::bitmap_t<>& behaviour_bitmap) {

  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::stop_watch_t watch;
  watch.mark("t0");
  watch.mark("t1");

  constexpr size_t mem_size = sizeof(plato::vid_t) * (size_t(std::numeric_limits<plato::vid_t>::max()) + 1);
  std::unique_ptr<plato::vid_t, plato::mmap_deleter> out_degree(
    (plato::vid_t*)mmap(nullptr, mem_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
    plato::mmap_deleter{mem_size});
  CHECK(MAP_FAILED != out_degree.get())
  << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno);

  plato::vid_t max_vid = 0;
  size_t edges = 0;
  {
    plato::thread_local_counter max_vid_counter;
    std::mutex mutex;
    std::vector<std::string> files = plato::get_files(FLAGS_input_behaviour_edges);

    plato::thread_local_buffer input_buffer;

    auto __send = [&] (plato::bsp_send_callback_t<plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>> send) {
      while (true) {
        std::string filename;
        {
          std::lock_guard<std::mutex> lock(mutex);
          if (files.empty()) break;
          filename = std::move(files.back());
          files.pop_back();
        }

        plato::with_file(filename, [&] (boost::iostreams::filtering_istream& is) {
          plato::vertex_csv_parser<boost::iostreams::filtering_istream, behaviour_state_content_t<sum_items_num>>(
            is,
            [&] (plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>* input, size_t size) {
              __sync_fetch_and_add(&edges, size);
              for (size_t i = 0; i < size; ++i) {
                plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>& v = input[i];
                send(get_behaviour_partition_id(v.vdata_.behaviour_id_), v);
              }
              return true;
            },
            [] (behaviour_state_content_t<sum_items_num>* output, char* s_input) {
              char* pSave  = nullptr;
              char* pLog = s_input;
              char* pToken = strtok_r(pLog, ", \t", &pSave);
              if (nullptr == pToken) {
                LOG(WARNING) << boost::format("can not extract behaviour id from (%s)") % pLog;
                return false;
              }
              output->behaviour_id_ = strtoul(pToken, nullptr, 10);

              for (unsigned i = 0; i < sum_items_num; ++i) {
                pLog = pToken;
                pToken = strtok_r(nullptr, ", \t", &pSave);
                if (nullptr == pToken) {
                  LOG(WARNING) << boost::format("can not extract sum_items[%u] from (%s)") % i % pLog;
                  return false;
                }
                output->sum_items_[i] = strtoul(pToken, nullptr, 10);
              }
              return true;
            }
          );
        });
      }
    };

    auto __recv = [&] (int /*p_i*/, plato::bsp_recv_pmsg_t<plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>>& pmsg) {
      plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>& msg = *pmsg;
      auto& counter = max_vid_counter.local();
      counter = std::max(counter, size_t(msg.vid_));
      __sync_fetch_and_add(out_degree.get() + msg.vid_, 1);
      behaviour_cache.push_back(msg);
      behaviour_bitmap.set_bit(msg.vid_);
    };

    auto rc = plato::fine_grain_bsp<plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>>(__send, __recv);
    CHECK(0 == rc) << "bsp failed with code: " << rc;

    max_vid = max_vid_counter.reduce_max();
    MPI_Allreduce(MPI_IN_PLACE, &max_vid, 1, plato::get_mpi_data_type<plato::vid_t>(), MPI_MAX, MPI_COMM_WORLD);
  }
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "load behaviour cache cost: " << watch.show("t1") / 1000.0 << "s";

  watch.mark("t1");
  size_t local_degree_sum = 0;
  size_t local_vertices = 0;
  #pragma omp parallel for reduction(+:local_degree_sum) reduction(+:local_vertices)
  for (plato::vid_t v_i = 0; v_i <= max_vid; v_i++) {
    auto degree = *(out_degree.get() + v_i);
    if (degree) {
      local_degree_sum += degree;
      local_vertices++;
      degree_unit v{v_i, degree};
      behaviour_degree_cache.push_back(v);
    }
  }
  size_t degree_sum = 0;
  MPI_Allreduce(&local_degree_sum, &degree_sum, 1, plato::get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &edges, 1, plato::get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
  CHECK(edges == degree_sum);
  behaviour_bitmap.sync();
  CHECK(behaviour_bitmap.msb() == max_vid);

  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "reduce behaviour out_degree & bitmap cost: " << watch.show("t1") / 1000.0 << "s, partition: " << cluster_info.partition_id_;
  LOG(INFO) << "local_vertices:   " << local_vertices;

  behaviour_graph_info.edges_    = edges;
  behaviour_graph_info.max_v_i_  = max_vid;
  behaviour_graph_info.vertices_ = behaviour_bitmap.count();

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << behaviour_graph_info.edges_;
    LOG(INFO) << "max_v_id:     " << behaviour_graph_info.max_v_i_;
    LOG(INFO) << "vertices:     " << behaviour_graph_info.vertices_;
    LOG(INFO) << "load behaviour cache & count out_degree & bitmap cost: " << watch.show("t0") / 1000.0 << "s";
  }
}

template <unsigned sum_items_num, template <typename, typename Enable = void> class OBJECT_BUFFER>
plato::sparse_state_t<behaviour_state_t<sum_items_num>, plato::empty_t> load_behaviour_graph_from_cache(
  OBJECT_BUFFER<plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>>&& behaviour_cache,
  plato::graph_info_t& behaviour_graph_info, OBJECT_BUFFER<degree_unit>&& behaviour_degree_cache) {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  plato::stop_watch_t watch;
  watch.mark("t0");
  watch.mark("t1");

  constexpr size_t mem_size = sizeof(plato::vid_t) * (size_t(std::numeric_limits<plato::vid_t>::max()) + 1);
  std::unique_ptr<plato::vid_t, plato::mmap_deleter> out_degree(
    (plato::vid_t*)mmap(nullptr, mem_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
    plato::mmap_deleter{mem_size});
  CHECK(MAP_FAILED != out_degree.get())
  << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno);

  size_t local_degree_sum = 0;
  plato::vid_t local_vertices = 0;
  plato::vid_t max_vid = 0;

  behaviour_degree_cache.reset_traversal();
  #pragma omp parallel reduction(+:local_degree_sum) reduction(+:local_vertices) reduction(max:max_vid)
  {
    size_t chunk_size = 1;
    while (behaviour_degree_cache.next_chunk([&] (size_t, degree_unit* v) {
      out_degree.get()[v->vid_] = v->degree_;
      local_degree_sum += v->degree_;
      local_vertices++;
      max_vid = std::max(max_vid, v->vid_);
    }, &chunk_size)) {}
  }

  size_t degree_sum = 0;
  MPI_Allreduce(MPI_IN_PLACE, &max_vid, 1, plato::get_mpi_data_type<plato::vid_t>(), MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_degree_sum, &degree_sum, 1, plato::get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
  CHECK(behaviour_graph_info.edges_ == degree_sum);
  CHECK(behaviour_graph_info.max_v_i_ == max_vid);

  plato::sparse_state_t<behaviour_state_t<sum_items_num>, plato::empty_t> behaviour(local_vertices * 1.2, std::make_shared<plato::empty_t>());

  {
    watch.mark("t1");

    behaviour_mmap_p.reset(
      mmap(nullptr, sizeof(behaviour_state_t<sum_items_num>) * local_degree_sum, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
      plato::mmap_deleter{sizeof(behaviour_state_t<sum_items_num>) * local_degree_sum});
    CHECK(MAP_FAILED != behaviour_mmap_p.get())
    << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno) << " local_degree_sum: " << local_degree_sum;

    behaviour.unlock();
    auto lock_defer = plato::defer([&]{ behaviour.lock(); });

    size_t adjs_num = 0;
    #pragma omp parallel for
    for (plato::vid_t v_i = 0; v_i <= max_vid; ++v_i) {
      plato::vid_t degree = *(out_degree.get() + v_i);
      if (degree) {
        behaviour.upsert(
          v_i, [] (behaviour_state_t<sum_items_num>&) { CHECK(false) << "duplicated vertex!"; },
          (behaviour_state_content_t<sum_items_num>*)behaviour_mmap_p.get() + __sync_fetch_and_add(&adjs_num, degree));
      }
    }
    CHECK(adjs_num == local_degree_sum) << boost::format("adjs_num: %lu, local_degree_sum: %lu") % adjs_num % local_degree_sum;
    out_degree.reset();
    MPI_Barrier(MPI_COMM_WORLD);

    LOG_IF(INFO, 0 == cluster_info.partition_id_) << "build behaviour index only table cost: " << watch.show("t1") / 1000.0 << "s";
  }

  watch.mark("t1");
  behaviour_cache.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 0;
    while (behaviour_cache.next_chunk([&] (size_t, const plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>* p) {
      const plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>& v = *p;
      behaviour_state_t<sum_items_num>& state = behaviour[v.vid_];
      state.behaviour_[__sync_fetch_and_add(&state.behaviour_size_, 1)] = v.vdata_;
    }, &chunk_size)) {}
  }

  {
    size_t behaviour_num = 0;
    behaviour.reset_traversal();
    #pragma omp parallel reduction(+:behaviour_num)
    {
      size_t chunk_size = 64;
      while (behaviour.next_chunk([&] (plato::vid_t /* v_i */, behaviour_state_t<sum_items_num>* state) {
        behaviour_num += state->behaviour_size_;
      }, &chunk_size)) { }
    }
    CHECK(behaviour_num == local_degree_sum) << "behaviour_num: " << behaviour_num << ", local_degree_sum: " << local_degree_sum;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "fill behaviour table cost: " << watch.show("t1") / 1000.0 << "s";
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "load behaviour tcsr cost: " << watch.show("t0") / 1000.0 << "s";
  return behaviour;
}

template <unsigned sum_items_num, template <typename, typename Enable = void> class OBJECT_BUFFER>
void spread() {
  auto& cluster_info = plato::cluster_info_t::get_instance();

  plato::stop_watch_t watch;
  watch.mark("t0");

  plato::object_buffer_opt_t behaviour_buffer_opt;
  behaviour_buffer_opt.path_ = FLAGS_cache_dir;
  behaviour_buffer_opt.prefix_ = (boost::format("behaviour-edge-%04ld-") % cluster_info.partition_id_).str();
  OBJECT_BUFFER<plato::vertex_unit_t<behaviour_state_content_t<sum_items_num>>> behaviour_cache(behaviour_buffer_opt);

  plato::graph_info_t behaviour_graph_info(true);

  plato::object_buffer_opt_t behaviour_degree_buffer_opt;
  behaviour_degree_buffer_opt.path_ = FLAGS_cache_dir;
  behaviour_degree_buffer_opt.prefix_ = (boost::format("behaviour-degree-%04ld-") % cluster_info.partition_id_).str();
  OBJECT_BUFFER<degree_unit> behaviour_degree_cache(behaviour_degree_buffer_opt);

  plato::bitmap_t<> behaviour_bitmap(size_t(std::numeric_limits<plato::vid_t >::max()) + 1);

  load_behaviour_graph_cache<sum_items_num, OBJECT_BUFFER>(behaviour_cache, behaviour_graph_info, behaviour_degree_cache, behaviour_bitmap);

  watch.mark("t1");
  plato::graph_info_t relations_graph_info(FLAGS_is_directed);
  OBJECT_BUFFER<relation_t> relations = load_realation_graph<OBJECT_BUFFER>(relations_graph_info, std::move(behaviour_bitmap));
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "load relations graph total cost: " << watch.show("t1") / 1000.0 << "s";

  watch.mark("t1");
  plato::sparse_state_t<behaviour_state_t<sum_items_num>, plato::empty_t> behaviour =
    load_behaviour_graph_from_cache<sum_items_num, OBJECT_BUFFER>(std::move(behaviour_cache), behaviour_graph_info, std::move(behaviour_degree_cache));
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "load behaviour graph from cache cost: " << watch.show("t1") / 1000.0 << "s";

  watch.mark("t1");
  {
    /******************* distributed  begin *******************/
    plato::thread_local_fs_output os(FLAGS_output, (boost::format("%04d_") % cluster_info.partition_id_).str(), true);
    plato::thread_local_counter distinct_uin;
    plato::thread_local_counter spread_counter;
    plato::thread_local_buffer list_buffer;
    plato::thread_local_buffer exist_uin_buffer;
    /******************* distributed  end *******************/

    g_adjs_buffer_p_.reset(new plato::thread_local_buffer);
    auto adjs_buffer_reset_defer = plato::defer([] { g_adjs_buffer_p_.reset(); });

    size_t size_thousand = relations_graph_info.vertices_ / 1000;
    size_t finished = 0;

    LOG(INFO) << boost::format("partition: %d, total num: %lu") % cluster_info.partition_id_ % relations_graph_info.vertices_;

    auto spread_func = [&] (relation_t& relation) {
      unsigned exist_uin_cnt = 0;
      auto* exist_uin_buf = (plato::vid_t*)exist_uin_buffer.local();

      unsigned size = 0;
      for (unsigned i = 0; i < relation.adjs_size_; ++i) {
        plato::vid_t v_i = relation.adjs_[i];
        behaviour.find_fn(v_i, [&] (behaviour_state_t<sum_items_num>& state) {
          CHECK(state.behaviour_size_);
          size += state.behaviour_size_;
          exist_uin_buf[exist_uin_cnt++] = v_i;
        });
      }

      if (size) {
        CHECK(exist_uin_cnt);
        distinct_uin.local()++;
        spread_counter.local() += size;
        boost::iostreams::filtering_stream<boost::iostreams::output>& local_os = os.local();

        if (FLAGS_output_list) {
          struct behaviour_result_t {
            uint32_t cnt_ = 0;
            unsigned offset_ = 0;
            uint32_t sum_items_[sum_items_num] = {};
          };

          google::dense_hash_map<int64_t, behaviour_result_t> mutual_map(size);
          mutual_map.set_empty_key(0);

          for (unsigned i = 0; i < exist_uin_cnt; ++i) {
            behaviour_state_t<sum_items_num>& state = behaviour[exist_uin_buf[i]];
            for (unsigned j = 0; j < state.behaviour_size_; ++j) {
              mutual_map[state.behaviour_[j].behaviour_id_].cnt_++;
            }
          }

          unsigned offset = 0;
          for (std::pair<const int64_t, behaviour_result_t>& pair : mutual_map) {
            pair.second.offset_ = offset;
            offset += pair.second.cnt_;
          }
          CHECK(offset == size);

          auto* list_buf = (plato::vid_t*)list_buffer.local();

          for (unsigned i = 0; i < exist_uin_cnt; ++i) {
            behaviour_state_t<sum_items_num>& state = behaviour[exist_uin_buf[i]];
            for (unsigned j = 0; j < state.behaviour_size_; ++j) {
              behaviour_state_content_t<sum_items_num>& behaviour = state.behaviour_[j];
              uint64_t behaviour_id = behaviour.behaviour_id_;
              behaviour_result_t& result = mutual_map[behaviour_id];
              result.cnt_++;
              list_buf[result.offset_++] = exist_uin_buf[i];
              for (unsigned k = 0; k < sum_items_num; k++) {
                result.sum_items_[k] += behaviour.sum_items_[k];
              }
            }
          }

          for (std::pair<const int64_t, behaviour_result_t>& pair : mutual_map) {
            behaviour_result_t& result = pair.second;

            local_os << relation.uin_ << "," << pair.first << ",";
            result.offset_ -= result.cnt_;

            for (unsigned i = result.offset_; i < result.cnt_ - 1; ++i) {
              local_os << list_buf[result.offset_ + i] << ":";
            }
            local_os << list_buf[result.offset_ + result.cnt_ - 1];

            for (unsigned i = 0; i < sum_items_num; i++) {
              local_os << "," << result.sum_items_[i];
            }
            local_os << std::endl;
          }
        } else {
          struct behaviour_result_t {
            uint32_t cnt_ = 0;
            uint32_t sum_items_[sum_items_num] = {};
          };

          google::dense_hash_map<int64_t, behaviour_result_t> mutual_map(size);
          mutual_map.set_empty_key(0);

          for (unsigned i = 0; i < exist_uin_cnt; ++i) {
            behaviour_state_t<sum_items_num>& state = behaviour[exist_uin_buf[i]];
            for (unsigned j = 0; j < state.behaviour_size_; ++j) {
              behaviour_state_content_t<sum_items_num>& behaviour = state.behaviour_[j];
              uint64_t behaviour_id = behaviour.behaviour_id_;
              behaviour_result_t& result = mutual_map[behaviour_id];
              result.cnt_++;
              for (unsigned k = 0; k < sum_items_num; k++) {
                result.sum_items_[k] += behaviour.sum_items_[k];
              }
            }
          }

          for (std::pair<const int64_t, behaviour_result_t>& pair : mutual_map) {
            behaviour_result_t& result = pair.second;
            local_os << relation.uin_ << "," << pair.first << "," << result.cnt_;
            for (unsigned i = 0; i < sum_items_num; i++) {
              local_os << "," << result.sum_items_[i];
            }
            local_os << std::endl;
          }
        }
      }

      size_t finished_ = __sync_fetch_and_add(&finished, 1);
      if ((finished_ % size_thousand == 0) && ((finished_ / size_thousand) % 10) == 0) {
        LOG(INFO) << boost::format("partition: %d, finished percent %lu%%, num: %lu") % cluster_info.partition_id_ % (finished_ / size_thousand / 10) % finished_;
      }
    };

    relations.reset_traversal();
    auto __send = [&] (plato::bsp_send_callback_t<relation_t> send) {
      size_t chunk_size = 0;
      while (relations.next_chunk([&] (size_t /* idx */, relation_t* p) {
        relation_t& relation = *p;
        for (int partition_id = 0; partition_id < cluster_info.partitions_; ++partition_id) {
          if (partition_id == cluster_info.partition_id_) {
            spread_func(relation);
          } else {
            send(partition_id, relation);
          }
        }
      }, &chunk_size)) { }
    };

    auto __recv = [&] (int /*p_i*/, plato::bsp_recv_pmsg_t<relation_t>& pmsg) {
      relation_t& relation = *pmsg;
      spread_func(relation);
    };

    plato::bsp_opts_t opts;
    opts.flying_send_per_node_ = std::max(cluster_info.threads_ / cluster_info.partitions_, 3);
    opts.flying_recv_ = cluster_info.threads_;
    auto rc = plato::fine_grain_bsp<relation_t>(__send, __recv, opts);
    CHECK(0 == rc) << "bsp failed with code: " << rc;
    CHECK(relations_graph_info.vertices_ == finished) << boost::format("total: %lu, finished: %lu.") % relations_graph_info.vertices_ % finished;

    size_t distinct_uin_total = distinct_uin.reduce_sum();
    size_t spread_counter_total = spread_counter.reduce_sum();

    MPI_Allreduce(MPI_IN_PLACE, &distinct_uin_total, 1, plato::get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &spread_counter_total, 1, plato::get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
    LOG_IF(INFO, 0 == cluster_info.partition_id_)
    << boost::format("distinct uin: %lu, spread result counter: %lu") % distinct_uin_total % spread_counter_total;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "spread & save cache cost: " << watch.show("t1") / 1000.0 << "s";
  LOG_IF(INFO, 0 == cluster_info.partition_id_) << "whole cost: " << watch.show("t0") / 1000.0 << "s";
}


template <unsigned sum_items_num>
void compute() {
  if (boost::istarts_with(FLAGS_cache_dir, "hdfs://")) {
    spread<sum_items_num, plato::object_dfs_buffer_t>();
  } else {
    spread<sum_items_num, plato::object_file_buffer_t>();
  }
}


int main(int argc, char** argv) {
  auto& cluster_info = plato::cluster_info_t::get_instance();
  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "input_relation_edges:      " << FLAGS_input_relation_edges;
    LOG(INFO) << "input_behaviour_edges:     " << FLAGS_input_behaviour_edges;
    LOG(INFO) << "output:                    " << FLAGS_output;
    LOG(INFO) << "output_list:               " << FLAGS_output_list;
    LOG(INFO) << "is_directed:               " << FLAGS_is_directed;
    LOG(INFO) << "split_factor:              " << FLAGS_split_factor;
    LOG(INFO) << "cache_dir:                 " << FLAGS_cache_dir;
    LOG(INFO) << "sum_items_num:             " << FLAGS_sum_items_num;
  }

  LOG(INFO) << "total memory:              " << (double)sysconf(_SC_PAGESIZE) * (double)sysconf(_SC_PHYS_PAGES) / 1024 / 1024 / 1024 << "GB";
  LOG(INFO) << "available memory:          " << (double)sysconf(_SC_PAGESIZE) * (double)sysconf(_SC_AVPHYS_PAGES) / 1024 / 1024 / 1024 << "GB";
  if (!boost::istarts_with(FLAGS_cache_dir, "hdfs://")) {
    boost::filesystem::create_directories(FLAGS_cache_dir);
    auto space = boost::filesystem::space(FLAGS_cache_dir);
    LOG(INFO) << "disk capacity:             " << space.capacity / 1024 / 1024 / 1024 << "GB";
    LOG(INFO) << "disk free:                 " << space.free / 1024 / 1024 / 1024 << "GB";
    LOG(INFO) << "disk available:            " << space.available / 1024 / 1024 / 1024 << "GB";
  }

  switch(FLAGS_sum_items_num) {
    case 0:
      compute<0>();
      break;
    case 1:
      compute<1>();
      break;
    case 2:
      compute<2>();
      break;
    case 3:
      compute<3>();
      break;
    case 4:
      compute<4>();
      break;
    case 5:
      compute<5>();
      break;
    default:
      LOG(ERROR) << "sum_items_num should <= 5" << FLAGS_sum_items_num;
      abort();
  }

  return 0;
}

