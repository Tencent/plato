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

#pragma once

namespace plato {

template<typename Func>
class deferred_action {
  Func _func;
  bool _cancelled = false;
public:
  static_assert(std::is_nothrow_move_constructible<Func>::value, "Func(Func&&) must be noexcept");

  deferred_action(Func &&func) noexcept : _func(std::move(func)) {}

  deferred_action(deferred_action &&o) noexcept : _func(std::move(o._func)), _cancelled(o._cancelled) {
    o._cancelled = true;
  }

  deferred_action &operator=(deferred_action &&o) noexcept {
    if (this != &o) {
      this->~deferred_action();
      new(this) deferred_action(std::move(o));
    }
    return *this;
  }

  deferred_action(const deferred_action &) = delete;

  ~deferred_action() { if (!_cancelled) { _func(); }; }

  void cancel() { _cancelled = true; }
};

template<typename Func>
inline
deferred_action<Func>
defer(Func &&func) {
  return deferred_action<Func>(std::forward<Func>(func));
}

}  // namespace plato

