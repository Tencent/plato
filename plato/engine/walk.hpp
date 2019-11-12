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

#ifndef __PLATO_ENGINE_WALK_HPP__
#define __PLATO_ENGINE_WALK_HPP__

#include <cstdint>
#include <cstdlib>

#include <queue>
#include <random>
#include <atomic>
#include <vector>
#include <limits>
#include <atomic>
#include <algorithm>
#include <type_traits>

#include "mpi.h"
#include "omp.h"
#include "glog/logging.h"
#include "libcuckoo/cuckoohash_map.hh"

#include "plato/util/hash.hpp"
#include "plato/util/perf.hpp"
#include "plato/util/buffer.hpp"
#include "plato/util/object_buffer.hpp"
#include "plato/util/aliastable.hpp"
#include "plato/util/mmap_alloc.hpp"
#include "plato/graph/graph.hpp"
#include "plato/graph/structure/hnbbcsr.hpp"
#include "plato/graph/structure/cbcsr.hpp"

namespace plato {

struct walk_opts_t {
  vid_t   max_steps_  = 0;     // maximum steps for one epoch
  vid_t   epochs_     = 0;     // how many rounds to walk
  double  start_rate  = 0.05;  // start 5% walker per one bsp, reduce memory consumption
};

void print_mem_info(std::string notice) {
  LOG(INFO) << "Notice: " << notice;
  mem_status_t mstat;
  self_mem_usage(&mstat);
  LOG(INFO) << "Current memory usage : " << (double)mstat.vm_rss / 1000.0 << "MBytes";
  LOG(INFO) << "Current memory peak : " << (double)mstat.vm_peak / 1000.0 << "MBytes";
  LOG(INFO) << "Current resident set peak : " << (double)mstat.vm_hwm / 1000.0 << "MBytes";
}

template <bool isnonweighted, typename STORAGE>
struct sample_traits { };

template <typename STORAGE>
struct sample_traits<false, STORAGE> {
  struct block_t {
    float  prob_;
    vid_t alias_;
  };

  struct __sampler_t {
    STORAGE* storage_;
    eid_t length_;
    std::unique_ptr<block_t[]> probs_;

    __sampler_t(STORAGE* storage) : storage_(storage), length_(storage->edges()), probs_(new block_t[length_]) {
      auto storage_index = storage->index();
      auto storage_adjs  = storage->adjs();
      for(size_t i = 0; i < (size_t)storage->non_zero_lines(); ++i) {
        eid_t start = storage_index.get()[i];
        eid_t end   = storage_index.get()[i + 1];
        float probs_sum = 0;
        eid_t len = end - start;
        eid_t small_count = 0;
        eid_t large_count = 0;
        std::unique_ptr<float>  norm_probs(new float[len]);
        std::unique_ptr<eid_t> large(new eid_t[len]);
        std::unique_ptr<eid_t> small(new eid_t[len]);
        for(eid_t j = start; j < end; ++j) {
          auto& nei = storage_adjs.get()[j];
          probs_sum += nei.edata_;
        }

        for(eid_t j = start; j < end; ++j) {
          auto& nei = storage_adjs.get()[j];
          norm_probs.get()[j - start] = nei.edata_ * len / probs_sum;
        }

        for (eid_t j = start; j < end; ++j) {
          eid_t offset = j - start;
          if (norm_probs.get()[offset] < 1.0) {
            small.get()[small_count++] = offset;
          } else {
            large.get()[large_count++] = offset;
          }
        }

        while (small_count && large_count) {
          eid_t small_idx = small.get()[--small_count];
          eid_t large_idx = large.get()[--large_count];

          probs_.get()[start + small_idx].prob_  = norm_probs.get()[small_idx];
          probs_.get()[start + small_idx].alias_ = large_idx;
  
          norm_probs.get()[large_idx] = norm_probs.get()[large_idx] - (1.0 - norm_probs.get()[small_idx]);

          if (norm_probs.get()[large_idx] < 1.0) {
            small.get()[small_count++] = large_idx;
          } else {
            large.get()[large_count++] = large_idx;
          }
        }

        while (large_count) {
          eid_t idx = large.get()[--large_count];
          probs_.get()[start + idx].prob_  = 1.0;
          probs_.get()[start + idx].alias_ = idx;
        }

        while (small_count) {  // can this happen ??
          eid_t idx = small.get()[--small_count];
          probs_.get()[start + idx].prob_  = 1.0;
          probs_.get()[start + idx].alias_ = idx;
        }
      }
    }

    __sampler_t(const __sampler_t&) = delete;
    __sampler_t& operator=(const __sampler_t&) = delete;

    template <typename URNG>
    typename STORAGE::adj_unit_spec_t* sample(vid_t v_i, URNG& g) {
      auto neis = storage_->neighbours(v_i);
      if((neis.begin_ == NULL && neis.end_ == NULL) || neis.begin_ == neis.end_) {
        return NULL;
      }

      size_t neighbour_count = neis.end_ - neis.begin_;
      std::uniform_real_distribution<float> dist1(0, 1.0);
      std::uniform_int_distribution<eid_t> dist2(0, neighbour_count - 1);
      eid_t k = dist2(g);
      eid_t global_offset = neis.begin_ - (storage_->adjs)().get();
      return dist1(g) < probs_.get()[global_offset + k].prob_ ? neis.begin_ + k : neis.begin_ + probs_.get()[global_offset + k].alias_;
    }

    template <typename URNG>
    typename STORAGE::adj_unit_spec_t* sample(vid_t v_i, size_t type, URNG& g) {
      auto neis = storage_->neighbours(v_i, type);
      if((neis.begin_ == NULL && neis.end_ == NULL) || neis.begin_ == neis.end_) {
        return NULL;
      }

      size_t neighbour_count = neis.end_ - neis.begin_;
      std::uniform_real_distribution<float> dist1(0, 1.0);
      std::uniform_int_distribution<eid_t> dist2(0, neighbour_count - 1);
      eid_t k = dist2(g);
      eid_t global_offset = neis.begin_ - (storage_->adjs)().get();
      return dist1(g) < probs_.get()[global_offset + k].prob_ ? neis.begin_ + k : neis.begin_ + probs_.get()[global_offset + k].alias_;
    }

    bool existed(vid_t v_i, vid_t target) {
      auto neis = storage_->neighbours(v_i);
      if((neis.begin_ == NULL && neis.end_ == NULL) || neis.begin_ == neis.end_) {
        return false;
      }
  
      auto* pt = neis.begin_;
      while(pt != neis.end_) {
        if(pt->neighbour_ == target) {
          return true;
        }
        ++pt;
      }
  
      return false;
    }
  };
  using type = __sampler_t;
};

template <typename STORAGE>
struct sample_traits<true, STORAGE> {
  struct __sampler_t {
    STORAGE* storage_;
    eid_t length_;

    __sampler_t(STORAGE* storage) : storage_(storage), length_(storage->edges()) {}
    __sampler_t(const __sampler_t&) = delete;
    __sampler_t& operator=(const __sampler_t&) = delete;

    template <typename URNG>
    typename STORAGE::adj_unit_spec_t* sample(vid_t v_i, URNG& g) {
      auto neis = storage_->neighbours(v_i);
      if((neis.begin_ == NULL && neis.end_ == NULL) || neis.begin_ == neis.end_) {
        return NULL;
      }

      size_t neighbour_count = neis.end_ - neis.begin_;
      std::uniform_int_distribution<vid_t> dist(0, neighbour_count - 1);
      return (neis.begin_ + dist(g));
    }

    template <typename URNG>
    typename STORAGE::adj_unit_spec_t* sample(vid_t v_i, size_t type, URNG& g) {
      auto neis = storage_->neighbours(v_i, type);
      if((neis.begin_ == NULL && neis.end_ == NULL) || neis.begin_ == neis.end_) {
        return NULL;
      }

      size_t neighbour_count = neis.end_ - neis.begin_;
      std::uniform_int_distribution<vid_t> dist(0, neighbour_count - 1);
      return (neis.begin_ + dist(g));
    }

    bool existed(vid_t v_i, vid_t target) {
      auto neis = storage_->neighbours(v_i);
      if((neis.begin_ == NULL && neis.end_ == NULL) || neis.begin_ == neis.end_) {
        return false;
      }

      auto* pt = neis.begin_;
      while(pt != neis.end_) {
        if(pt->neighbour_ == target) {
          return true;
        }
        ++pt;
      }

      return false;
    }
  };
  using type = __sampler_t;
};

struct footprint_t {
  footprint_t(size_t n, vid_t v_i)
    : idx_(0), path_(new vid_t[n]) {
    push_back(v_i);
  }

  footprint_t(void): idx_(0), path_(nullptr) { }

  footprint_t& operator=(footprint_t&& x) {
    idx_   = x.idx_;
    path_  = x.path_;

    x.idx_  = 0;
    x.path_ = nullptr;

    return *this;
  }

  footprint_t(footprint_t&& x) {
    this->operator=(std::forward<footprint_t>(x));
  }

  footprint_t(const footprint_t&) = delete;
  footprint_t& operator=(const footprint_t&) = delete;

  ~footprint_t(void) {
    if (nullptr != path_) { delete[] path_; }
  }

  void push_back(vid_t v_i) {
    path_[idx_] = v_i;
    idx_ += 1;
  }

  vid_t  idx_;
  vid_t* path_;
};

using footprint_storage_t = cuckoohash_map<vid_t, footprint_t, cuckoo_vid_hash>;

template <typename DATA>
struct walker_t {
  DATA udata_;  // user data

  // ******************************************************************************* //
  // system will fill below fields

  vid_t walk_id_;       // path-id
  vid_t step_id_;       // step-id, start from 0
  vid_t current_v_id_;  // current step vertex's id

  // ******************************************************************************* //

};

template <typename DATA, typename PART_IMPL>
struct walk_context_t {
  using walker_spec_t = walker_t<DATA>;
  using sender_t = bsp_send_callback_t<walker_spec_t>;

  const walk_opts_t&      opts_;
  sender_t&               sender_;
  PART_IMPL&              partitioner_;
  footprint_storage_t&    footprints_;
  std::mt19937&           urng_;

  // a thread-safe uniform random generator
  std::mt19937& urng(void) { return urng_; }

  /*
   * move walker to a new vertex
   *
   * \param v_i     vertex-id to send to
   * \param walker  walker wait to move
   *
   * */
  void move_to(vid_t v_i, const walker_spec_t& walker) {
    sender_(partitioner_.get_partition_id(v_i), walker);

    // because of ordering relation exists between steps(one-bsp),
    // target-id optimizations is tricky.
  }

  // keep footprint in current partition 
  void keep_footprint(const walker_spec_t& walker) {
    footprints_.upsert(walker.walk_id_,
      [&](footprint_t& fp) {
        CHECK(fp.idx_ < opts_.max_steps_) << "walk_id_: " << walker.walk_id_ << ", step_id_: " << walker.step_id_
          << ", idx_: " << fp.idx_ << ", max_steps_: " << opts_.max_steps_;
        fp.push_back(walker.current_v_id_);
      },
      opts_.max_steps_, walker.current_v_id_);
  }

  footprint_t erase_footprint(const walker_spec_t& walker) {
    footprint_t footprint;
    footprints_.erase_fn(walker.walk_id_, [&](footprint_t& fp) {
      footprint = std::move(fp);
      return true;
    });
    return footprint;
  }
};

template <typename STORAGE, typename EDATA, typename PART_IMPL = hash_by_source_t<>>
class walk_engine_t {
public:

  using partition_t = PART_IMPL;
  using adj_unit_list_spec_t = plato::adj_unit_list_t<EDATA>;
  using sample_state_t = typename sample_traits<std::template is_empty<EDATA>::value, STORAGE>::type;
  //using sample_state_t = sample_traits<false, STORAGE>;

  /*
   * construct an engine for normal walk
   **/

  /*
   * construct an engine for metapath-based walk
   *
   * \tparam EDGE_CACHE  edge's cache
   * \tparam V_TYPE      vertex's type storage, like dense_state_t<uint32_t, ...>
   *
   **/
   template <typename EDGE_CACHE>
   walk_engine_t(const graph_info_t& graph_info, EDGE_CACHE& cache, std::shared_ptr<partition_t> part);

  template <typename EDGE_CACHE, typename V_TYPE>
  walk_engine_t(const graph_info_t& graph_info, EDGE_CACHE& cache,
      V_TYPE& v_types, std::shared_ptr<partition_t> part);

  ~walk_engine_t() { delete storage_; delete sampler_; };
  /*
   * metapath-based, distinguish vertex's type
   **/
  // template <typename EDGE_CACHE, typename V_STATE>
  // walk_engine_t(EDGE_CACHE& cache, V_STATE& v_state);

  walk_engine_t(const walk_engine_t&) = delete;
  walk_engine_t& operator=(const walk_engine_t&) = delete;

  /*
   * set all vertex's upper-bound
   *
   * \tparam F     upperbound set callback functor. It should implement the method
   *               <tt>float operator()(vid_t v_i, state_t* state)<\tt>
   *
   * \param func   upperbound set callback functor
   **/
  template <typename F>
  void set_upperbound(F&& func);

  /*
   * perform random-walk, choose next step by current step's state
   *
   * \tparam DATA         extra data bind with walker
   *
   * \tparam INIT_WALK    walker initialize functor, should implement the method
   *                      <tt>void(walker_t*)<\tt>
   * 
   * \tparam WALK         move a walker to a new place, should implement the method
   *                      <tt>void(walk_context_t&&, walker_t<DATA>& walker)<\tt>
   *
   **/
   
  template <typename DATA, typename INIT_WALK, typename WALK>
  void walk(INIT_WALK&& init_walker, WALK&& take_walk, const walk_opts_t& opts);

  STORAGE* storage(void)                              { return storage_;     }

  sample_state_t* sampler(void)                        { return sampler_;    }
  // get partitioner of this walk engine
  std::shared_ptr<partition_t> partitioner(void)      { return partitioner_; }

protected:
  graph_info_t                              graph_info_;
  
  STORAGE*                                  storage_;

  sample_state_t*                           sampler_;

  std::shared_ptr<partition_t>              partitioner_;

  template <typename... Args>
  void INFO_0(const char* f, Args&&... args) {
    if (0 == cluster_info_t::get_instance().partition_id_) {
      int size = snprintf(nullptr, 0, f, args...);
      std::string res;
      res.resize(size);
      snprintf(&res[0], size + 1, f, args...);
      LOG(INFO) << res;
    }
  }
};

template <typename STORAGE, typename EDATA, typename PART_IMPL>
template <typename EDGE_CACHE>
walk_engine_t<STORAGE, EDATA, PART_IMPL>::walk_engine_t(const graph_info_t& graph_info, EDGE_CACHE& cache, std::shared_ptr<partition_t> part)
  : graph_info_(graph_info), partitioner_(part) {
  
  plato::stop_watch_t watch;
  auto& cluster_info = cluster_info_t::get_instance();
  CHECK((std::is_same<STORAGE, plato::cbcsr_t<EDATA, partition_t>>::value));
  watch.mark("t0");
  storage_ = new plato::cbcsr_t<EDATA, partition_t>(partitioner_);
  CHECK(storage_ != NULL);
  storage_->load_from_cache(graph_info, cache);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "Graph loading time :" << watch.show("t0") / 1000.0 << "s";
  }
  
  sampler_ = new sample_state_t(storage_);
  CHECK(sampler_ != NULL);
}

template <typename STORAGE, typename EDATA, typename PART_IMPL>
template <typename EDGE_CACHE, typename V_TYPE>
walk_engine_t<STORAGE, EDATA, PART_IMPL>::walk_engine_t(const graph_info_t& graph_info, EDGE_CACHE& cache,
    V_TYPE& v_types, std::shared_ptr<partition_t> part)
  : graph_info_(graph_info), partitioner_(part) {

  plato::stop_watch_t watch;
  auto& cluster_info = cluster_info_t::get_instance();
  CHECK((std::is_same<STORAGE, plato::hnbbcsr_t<EDATA, partition_t>>::value));

  watch.mark("t0");
  storage_ = new plato::hnbbcsr_t<EDATA, partition_t>(partitioner_);
  CHECK(storage_ != NULL);
  
  storage_->load_from_cache(graph_info, cache, v_types);
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "Graph loading time :" << watch.show("t0") / 1000.0 << "s";
  }
  LOG(INFO) << "start initial sampler......";
  sampler_ = new sample_state_t(storage_);
  LOG(INFO) << "initial sampler end......";
  CHECK(sampler_ != NULL);
}

template <typename STORAGE, typename EDATA, typename PART_IMPL>
template <typename DATA, typename INIT_WALK, typename WALK>
void walk_engine_t<STORAGE, EDATA, PART_IMPL>::walk(INIT_WALK&& init_walker, WALK&& take_walk, const walk_opts_t& opts) {
  using walker_spec_t       = walker_t<DATA>;
  using walk_context_spec_t = walk_context_t<DATA, PART_IMPL>;
  
  //print_mem_info("Before memory footprint");
  // path-storage
  footprint_storage_t footprints(storage_->non_zero_lines() * 1.5);
  plato::stop_watch_t watch;

  bsp_opts_t bsp_opts;
  bsp_opts.global_size_    = 64 * MBYTES;
  bsp_opts.local_capacity_ = 8 * PAGESIZE;

  bsp_buffer_t current_walkers;
  bsp_buffer_t next_walkers;

  watch.mark("t0");
  for (plato::vid_t e_i = 0; e_i < opts.epochs_; ++e_i) {
    plato::vid_t started_walkers   = 0;
    plato::vid_t total_walkers     = storage_->non_zero_lines();
    plato::vid_t walkers_per_round = (plato::vid_t)(total_walkers * opts.start_rate) + 1 ;

    size_t actives = 0;
    footprints.clear();
    {
      plato::traverse_opts_t opts;
      opts.mode_ = plato::traverse_mode_t::ORIGIN;
      storage_->reset_traversal(opts);
    }

    watch.mark("t1");
    for (plato::vid_t s_i = 0; 0 != actives || 0 == s_i; ++s_i) {
      watch.mark("t2");

      bsp_chunk_vistor_t walker_vistor(current_walkers);
      size_t expected_walkers = std::min(started_walkers + walkers_per_round, total_walkers);
    
      actives = 0;
      auto send_task = [&](plato::bsp_send_callback_t<walker_spec_t> send) {
        std::mt19937 urng(std::random_device{}());
        size_t __actives = 0;
        bool is_start_sucess = true;
        
        while (started_walkers < expected_walkers && is_start_sucess) {  // start new walker first
          size_t chunk_size = 1;
          
          is_start_sucess = storage_->next_chunk([&](plato::vid_t v_i, const adj_unit_list_spec_t& ) {
            walker_spec_t walker;
            walker.walk_id_      = v_i;
            walker.step_id_      = 0;
            walker.current_v_id_ = v_i;

            init_walker(&walker);
            take_walk(walk_context_spec_t { opts, send, *partitioner_, footprints, urng }, walker);
            ++__actives;
            __sync_fetch_and_add(&started_walkers, 1);
            return true;
          }, &chunk_size);
        }
        
        while (walker_vistor.next_chunk<walker_spec_t>([&](int p_i, bsp_recv_pmsg_t<walker_spec_t>& pwalker) {
          auto& walker = *pwalker;
          take_walk(walk_context_spec_t { opts, send, *partitioner_, footprints, urng },
              walker);
          ++__actives;
        })) { }
        __sync_fetch_and_add(&actives, __actives);
      };

      bsp<walker_spec_t>(&next_walkers, send_task, bsp_opts);
      
      vid_t gstarted_walkers = 0;
      MPI_Allreduce(&started_walkers, &gstarted_walkers, 1, get_mpi_data_type<decltype(started_walkers)>(), MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &actives, 1, get_mpi_data_type<decltype(actives)>(), MPI_SUM, MPI_COMM_WORLD);

      INFO_0("epoch[%04u], step[%04u], started[%010u], actives[%012lu], cost: %lfs", e_i, s_i, gstarted_walkers, actives,
          watch.show("t2") / 1000.0);

#ifdef __WALK_DEBUG__
      {
        mem_status_t mstat;
        self_mem_usage(&mstat);
        LOG(INFO) << "memory footprint: " << (double)mstat.vm_rss / 1000.0 << "MBytes";
      }
      
      LOG(INFO) << "retained footprints: " << footprints.size();
#endif
      
      swap(current_walkers, next_walkers);
    }

    INFO_0("epoch[%04u] done, cost: %lfs", e_i, watch.show("t1") / 1000.0);
  }
  //print_mem_info("check memory footprint");
  INFO_0("random walk done, cost: %lfs", watch.show("t0") / 1000.0);
}

// ************************************************************************************ //

}  // namespace plato

#endif

