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

#ifndef __PLATO_GRAPH_PARTITION_DUMMY_HPP__
#define __PLATO_GRAPH_PARTITION_DUMMY_HPP__

#include <cstdint>
#include <cstdlib>
#include <functional>

#include "plato/graph/base.hpp"

namespace plato {

class dummy_part_t {
public:

  // ******************************************************************************* //
  // required types & methods

  static constexpr graph_info_mask_t needed_graph_info(void) { return 0; }

  // get edge's partition
  int get_partition_id(vid_t /*src*/, vid_t /*dst*/) { return 0; }

  // get vertex's partition
  int get_partition_id(vid_t /*v_i*/) { return 0; }

  dummy_part_t(const graph_info_t& /*graph_info*/) { }

  // ******************************************************************************* //

protected:
};

}  // namespace plato

#endif

