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

#ifndef __PLATO_GRAPH_STATE_DETAIL_HPP__
#define __PLATO_GRAPH_STATE_DETAIL_HPP__

#include <cstdint>
#include <cstdlib>

#include "omp.h"
#include "mpi.h"

#include <memory>
#include <functional>

#include "plato/graph/base.hpp"
#include "plato/parallel/mpi.hpp"

namespace plato {

/*
 * process all/subset of vertices
 *
 * \param process     user define callback for each eligible vertex
 *                     R(vid_t v_i, value_t* value)
 * \param bitmap      bitmap used for filter subset of the vertex
 * \param chunk_size  at most process 'chunk_size' chunk at a batch
 *
 * \return sum of 'process' return
 **/
template <typename R, typename STATE, typename PROCESS>
R __foreach (
    STATE* state,
    PROCESS&& process,
    typename STATE::bitmap_spec_t* bitmap = nullptr,
    size_t chunk_size = PAGESIZE) {

  using value_t       = typename STATE::value_t;
  using bitmap_spec_t = typename STATE::bitmap_spec_t;

  R r_reduce = R();
  std::shared_ptr<bitmap_spec_t> p_bitmap(bitmap, [](bitmap_spec_t*) { });
  auto& cluster_info = cluster_info_t::get_instance();

  state->reset_traversal(p_bitmap);
  #pragma omp parallel reduction(+:r_reduce) num_threads(cluster_info.threads_)
  {
    size_t __chunk_size = chunk_size;
    R l_reduce = R();
    while (state->next_chunk([&](vid_t v_i, value_t* pval) {
      l_reduce += process(v_i, pval);
      return true;
    }, &__chunk_size)) { }

    r_reduce += l_reduce;
  }

  R g_reduce = R();
  MPI_Allreduce(&r_reduce, &g_reduce, 1, get_mpi_data_type<R>(), MPI_SUM, MPI_COMM_WORLD);
  return g_reduce;
}

}

#endif

