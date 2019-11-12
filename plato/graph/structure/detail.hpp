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

#ifndef __PLATO_GRAPH_STRUCTURE_DETAIL_HPP__
#define __PLATO_GRAPH_STRUCTURE_DETAIL_HPP__

#include <functional>

namespace plato { namespace structure_detail {

template <typename T>
struct disarmable_deleter_t {
  bool armed_;
  std::function<void(T*)> deleter_;

  void operator()(T* p) { if (armed_) { deleter_(p); } }

  template <typename Deleter>
  disarmable_deleter_t(Deleter deleter)
    : armed_(true), deleter_(deleter) { }

  template <typename Deleter>
  void reset_deleter(Deleter deleter) {
    deleter_ = deleter;
  }
};

}}

#endif

