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

#ifndef __PLATO_GRAPH_PARTITION_HASH_HPP__
#define __PLATO_GRAPH_PARTITION_HASH_HPP__

#include <cstdint>
#include <cstdlib>
#include <functional>

#include "plato/graph/base.hpp"

namespace plato {

template <typename Hash = std::hash<vid_t>>
class hash_by_source_t {
public:
  // ******************************************************************************* //
  // required types & methods

  /*
   * get edge's partition
   *
   * \param src  source vertex id
   * \param dst  destination vertex id
   *
   * \return partition_id
   **/
  int get_partition_id(vid_t src, vid_t /*dst*/) {
    return (int)(hfunc_(src) % (vid_t)partitions_);
  }

  /*
   * get vertex's partition
   *
   * \param v_i  vertex id
   *
   * \return partition_id
   **/
  int get_partition_id(vid_t v_i) {
    return (int)(hfunc_(v_i) % (vid_t)partitions_);
  }

  // ******************************************************************************* //

  hash_by_source_t(const Hash& hfunc = Hash())
    : hfunc_(hfunc), partitions_(0), partition_id_(0) {
    auto& cluster_info = cluster_info_t::get_instance();

    partitions_   = cluster_info.partitions_;
    partition_id_ = cluster_info.partition_id_;
  }

protected:
  Hash hfunc_;  // hash functor
  int  partitions_;
  int  partition_id_;
};

template <typename Hash = std::hash<vid_t>>
class hash_by_destination_t {
public:
  // ******************************************************************************* //
  // required types & methods

  /*
   * get edge's partition
   *
   * \param src  source vertex id
   * \param dst  destination vertex id
   *
   * \return partition_id
   **/
  int get_partition_id(vid_t /*src*/, vid_t dst) {
    return (int)(hfunc_(dst) % (vid_t)partitions_);
  }

  /*
   * get vertex's partition
   *
   * \param v_i  vertex id
   *
   * \return partition_id
   **/
  int get_partition_id(vid_t v_i) {
    return (int)(hfunc_(v_i) % (vid_t)partitions_);
  }

  // ******************************************************************************* //

  hash_by_destination_t(const Hash& hfunc = Hash())
    : hfunc_(hfunc), partitions_(0), partition_id_(0) {
    auto& cluster_info = cluster_info_t::get_instance();

    partitions_   = cluster_info.partitions_;
    partition_id_ = cluster_info.partition_id_;
  }

protected:
  Hash hfunc_;  // hash functor
  int  partitions_;
  int  partition_id_;
};

}  // namespace plato

#endif

