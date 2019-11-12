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

#ifndef __PLATO_GRAPH_STATE_VIEW_HPP__
#define __PLATO_GRAPH_STATE_VIEW_HPP__

#include <cstdint>
#include <cstdlib>

#include <memory>
#include <utility>
#include <functional>
#include <type_traits>

#include "glog/logging.h"

#include "plato/graph/base.hpp"
#include "plato/util/bitmap.hpp"

namespace plato {

template <typename VIEW, typename BITMAP>
struct active_v_view {
  VIEW   view_;
  BITMAP bitmap_;

  // ******************************************************************************* //
  // required types & methods

  // traverse related
  inline void reset_traversal(const traverse_opts_t& opts = traverse_opts_t());

  // Func looks like void(vid_t v_i, args...)
  template <typename Func>
  inline bool next_chunk(Func&& traversal, size_t* chunk_size);

  /*
   * foreach elements in this view, run in parallel
   *
   * \tparam PROCESS    R(vid_t v_i, args...)
   *
   * \param  traversal  process callback
   *
   * \return
   *      sum of process return
   **/
  template <typename R, typename PROCESS>
  R foreach(PROCESS&& traversal);

  // ******************************************************************************* //
};

template <typename VIEW, typename BITMAP>
inline active_v_view<VIEW, BITMAP> create_active_v_view(VIEW&& view, BITMAP&& bitmap) {
  return { std::forward<VIEW>(view), std::forward<BITMAP>(bitmap) };
}

// ************************************************************************************ //
// implementations

template <typename VIEW, typename BITMAP>
void active_v_view<VIEW, BITMAP>::reset_traversal(const traverse_opts_t& opts) {
  view_.reset_traversal(opts);
}

namespace view_detail {

template <typename F, typename BITMAP>
struct traversal_rebind_t {
  F func_; BITMAP bitmap_;

  template <typename... Args>
  inline void operator()(vid_t v_i, Args... args) const {
    if (bitmap_.get_bit(v_i)) {
      func_(v_i, std::forward<Args>(args)...);
    }
  }
};

template <typename F, typename BITMAP>
traversal_rebind_t<F, BITMAP>
bind_traversal(F&& func, BITMAP&& bitmap) {
  return { std::forward<F>(func), std::forward<BITMAP>(bitmap) };
}

template <typename R, typename F, typename BITMAP>
struct reduction_rebind_t {
  R&& rdu_; F func_; BITMAP bitmap_;

  template <typename... Args>
  inline void operator()(vid_t v_i, Args... args) {
    if (bitmap_.get_bit(v_i)) {
      rdu_ += func_(v_i, std::forward<Args>(args)...);
    }
  }
};

template <typename R, typename F, typename BITMAP>
reduction_rebind_t<R, F, BITMAP>
bind_reduction(R&& rdu, F&& func, BITMAP&& bitmap) {
  return { std::forward<R>(rdu), std::forward<F>(func), std::forward<BITMAP>(bitmap) };
}

}

template <typename VIEW, typename BITMAP>
template <typename Func>
bool active_v_view<VIEW, BITMAP>::next_chunk(Func&& traversal, size_t* chunk_size) {
  return view_.next_chunk(view_detail::bind_traversal(std::forward<Func>(traversal), std::forward<BITMAP>(bitmap_)), chunk_size);
}

template <typename VIEW, typename BITMAP>
template <typename R, typename PROCESS>
R active_v_view<VIEW, BITMAP>::foreach(PROCESS&& traversal) {
  R rdu = R();

  reset_traversal();
  #pragma omp parallel reduction(+:rdu)
  {
    R __rdu = R();
    size_t chunk_size = 4 * PAGESIZE;
    auto __traversal = view_detail::bind_reduction(std::forward<R>(__rdu), std::forward<PROCESS>(traversal),
        std::forward<BITMAP>(bitmap_));
    while (view_.next_chunk(__traversal, &chunk_size)) {  }
    rdu += __traversal.rdu_;
  }

  R grdu = R();
  MPI_Allreduce(&rdu, &grdu, 1, get_mpi_data_type<R>(), MPI_SUM, MPI_COMM_WORLD);
  return grdu;
}

// ************************************************************************************ //

}

#endif

