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

#ifndef __PLATO_PARALLER_STREAM_HPP__
#define __PLATO_PARALLER_STREAM_HPP__

#include <cstdint>
#include <cstdlib>

#include <vector>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "plato/graph/base.hpp"

namespace plato {

template <typename MSG>
using stream_source_t = std::function<MSG(void)>;
using stream_task_t   = std::function<void(pipe_in, pipe_out)>;

struct stream_opts_t {
  int threads_                = -1;
  int flying_events_per_node_ = -1;
};

namespace stream_detail {

}

template <typename MSG>
struct stream_event_t {
  int node_id_;
  MSG message_;
};

template <typename MSG>
struct stream_message_t {

};

struct stream_context_t {

};

template <typename , typename >
struct __stream_task_t {

  

};

/*
 * high level communication abstraction, stream
 *
 * \tparam SOURCE  
 *
 *                  <tt>stream_event_t(void)<\tt>
 *
 * \tparam Tasks
 *
 *                  <tt>void(MSG, stream_context_t)<\tt>
 *
 *
 **/
template <typename SOURCE, typename... Tasks>
void minibatch_stream(SOURCE&& source, Tasks&&... tasks) {







}


class stream_t {
public:

  template <typename SOURCE, typename... Tasks>
  stream_t(SOURCE&& source, Tasks&&... tasks);

  stream_t(const stream_t&) = delete;
  stream_t& operator=(const stream_t&) = delete;

  void regist_source(stream_source_t&& source);
  void regist_task(uint32_t cmd, stream_task_t&& task);
  void run(void);

protected:


};

// ************************************************************************************ //
// implementations




// ************************************************************************************ //

}

#endif

