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

#ifndef __PLATO_UTIL_ATOMIC_HPP__
#define __PLATO_UTIL_ATOMIC_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <assert.h>

namespace plato {

template <class T>
inline bool cas(T * ptr, T old_val, T new_val) {
  if (sizeof(T) == 8) {
    return __sync_bool_compare_and_swap((long*)ptr, *((long*)&old_val), *((long*)&new_val));
  } else if (sizeof(T) == 4) {
    return __sync_bool_compare_and_swap((int*)ptr, *((int*)&old_val), *((int*)&new_val));
  } else {
    assert(false);
  }
}

// to make warning go away
// warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
template <>
inline bool cas<uint32_t>(uint32_t* ptr, uint32_t old_val, uint32_t new_val) {
  return __sync_bool_compare_and_swap(ptr, old_val, new_val);
}
template <>
inline bool cas<int32_t>(int32_t* ptr, int32_t old_val, int32_t new_val) {
  return __sync_bool_compare_and_swap(ptr, old_val, new_val);
}
template <>
inline bool cas<uint64_t>(uint64_t* ptr, uint64_t old_val, uint64_t new_val) {
  return __sync_bool_compare_and_swap(ptr, old_val, new_val);
}
template <>
inline bool cas<int64_t>(int64_t* ptr, int64_t old_val, int64_t new_val) {
  return __sync_bool_compare_and_swap(ptr, old_val, new_val);
}

template <class T>
inline bool write_min(T * ptr, T val) {
  volatile T curr_val;
  bool done = false;
  do {
    curr_val = *ptr;
  } while (curr_val > val && !(done = cas(ptr, curr_val, val)));
  return done;
}

template <class T>
inline bool write_max(T * ptr, T val) {
  volatile T curr_val;
  bool done = false;
  do {
    curr_val = *ptr;
  } while (curr_val < val && !(done = cas(ptr, curr_val, val)));
  return done;
}

template <class T>
inline void write_add(T * ptr, T val) {
  volatile T new_val, old_val;
  do {
    old_val = *ptr;
    new_val = old_val + val;
  } while (!cas(ptr, old_val, new_val));
}

}  // namespace plato

#endif
