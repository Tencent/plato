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

#ifndef __PLATO_UTIL_hyperloglog_HPP__
#define __PLATO_UTIL_hyperloglog_HPP__

#include "plato/util/hash.hpp"
#include <array>
#include <cmath>
#include <cstring>

namespace plato {

template<uint32_t P = 12>
struct hyperloglog_t {

  static constexpr uint32_t HLL_P = P;
  static constexpr uint32_t HLL_REGISTERS = (1 << P);
  static constexpr uint32_t HLL_Q = (32 - P);
  uint8_t registers_[HLL_REGISTERS];

  /**
   * @brief init
   */
  void init() {
    memset(registers_, 0, sizeof(registers_));
  }

  /**
   * @brief
   * @tparam T
   * @param buff
   * @param len
   */
  template <typename T>
  void add(const T* buff, uint32_t len) {
    auto get_clz = [] (uint32_t hash, uint32_t bits) {
      uint8_t v = 1;
      while (v <= bits && !(hash & 0x80000000)) {
        v++;
        hash <<= 1;
      }
      return v;
    };

    uint32_t hash = murmurhash3(buff, len, 0x5f61767a);
    uint32_t index = hash >> HLL_Q;
    uint8_t rank = get_clz(hash << HLL_P, HLL_Q);
    if(rank > registers_[index]) {
      registers_[index] = rank;
    }
  }

  /**
   * @brief
   * @return
   */
  double estimate() const {
    double alpha_mm;
    uint32_t i;
    switch(HLL_P) {
      case 4:
        alpha_mm = 0.673;
        break;
      case 5:
        alpha_mm = 0.697;
        break;
      case 6:
        alpha_mm = 0.709;
        break;
      default:
        alpha_mm = 0.7213 / (1.0 + 1.079 / static_cast<double>(HLL_REGISTERS));
        break;
    }
    alpha_mm *= (static_cast<double>(HLL_REGISTERS) * static_cast<double>(HLL_REGISTERS));
    double sum = 0;
    for(i = 0; i < HLL_REGISTERS; i++) {
      sum += 1.0 / (1 << registers_[i]);
    }
    double estimate = alpha_mm / sum;
    if(estimate <= 5.0 / 2.0 * static_cast<double>(HLL_REGISTERS)) {
      int zeros = 0;
      for(i = 0; i < HLL_REGISTERS; i++) {
        zeros += ((this->registers_)[i] == 0);
      }
      if(zeros) {
        estimate = static_cast<double>(HLL_REGISTERS) * std::log(static_cast<double>(HLL_REGISTERS) / zeros);
      }
    }
    else if (estimate > (1.0 / 30.0) * 4294967296.0) {
      estimate = -4294967296.0 * log(1.0 - (estimate / 4294967296.0));
    }

    return estimate;
  }

  /**
   * @brief
   * @param hll
   * @return
   */
  int merge(const hyperloglog_t& hll) {
    int change = 0;
    if(HLL_P != hll.HLL_P) {
      return -1;
    }

    for(uint32_t i = 0; i < hll.HLL_REGISTERS; ++i) {
      if((hll.registers_)[i] > (registers_)[i]) {
        (registers_)[i] = (hll.registers_)[i];
        ++change;
      }
    }

    return change;
  }

  /**
   * @brief
   */
  void clear() {
    memset(registers_, 0, sizeof(registers_));
  }
}__attribute__((__packed__));
}

#endif
