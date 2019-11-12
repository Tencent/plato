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

#ifndef __PLATO_ALIASTABLE_HPP__
#define __PLATO_ALIASTABLE_HPP__

#include <cstdint>
#include <cstring>

#include <stack>
#include <random>
#include <memory>
#include <utility>
#include <algorithm>
#include <type_traits>

#include "glog/logging.h"

namespace plato {

namespace aliastable_detail {

template <typename T, typename PROB>
static typename std::enable_if<std::is_floating_point<T>::value, PROB&>::type
prob_traits(T& v) { return v; }

template <typename T, typename PROB>
static typename std::enable_if<!std::is_floating_point<T>::value, PROB&>::type
prob_traits(T& v) { return v.prob_; }

}

/*
 * Alias Method used for O(1) sampling
 *
 * references:
 *  [1] http://www.keithschwarz.com/darts-dice-coins/
 *  [2] Michael Vose. Linear Algorithm For Generating Random Numbers With a Given Distribution
 *
 * \tparam  PROB   the type of probability, can be float or double
 * \tparam  INDEX  the type of index, can be any integral type
 **/
template <typename PROB = float, typename INDEX = uint32_t>
class alias_table_t {
public:
  struct block_t {
    PROB  prob_;
    INDEX alias_;
  };

  static_assert(std::is_floating_point<PROB>::value, "PROB can only be floating type");
  static_assert(std::is_integral<INDEX>::value, "INDEX can only be integral type");

  // build alias table directly from probs, you needn't call initialize separately
  alias_table_t(PROB* probs, INDEX length);

  // reserve space for alias table, user should write probs direct into probs_
  // then call initialize to build alias table
  alias_table_t(INDEX length);

  alias_table_t(alias_table_t&&);

  alias_table_t(const alias_table_t&) = delete;
  alias_table_t& operator=(const alias_table_t&) = delete;

  void initialize(void);  // initialize from probs_

  // return how many elements in this table
  INDEX size(void);

  // resize this table and copy first std::min(n, length_) elements
  // to its new location.
  // after resize, you may want to reinitialize it.
  void resize(size_t n);

  // access probs_[ith]
  block_t& operator[] (INDEX i);

  /*
   * \tparam URNG  the type of uniform random number generator
   *
   * \param  g     A uniform random number generator object,
   *               used as the source of randomness.
   *
   * \return sampled idx
   * */
  template <typename URNG>
  inline INDEX sample(URNG& g);

protected:
  INDEX length_;
  std::unique_ptr<block_t[]> probs_;

  template <typename T>
  void initialize(T* probs);
};

// ************************************************************************************ //
// implementations

template <typename PROB, typename INDEX>
alias_table_t<PROB, INDEX>::alias_table_t(PROB* probs, INDEX length)
  : length_(length), probs_(new block_t[length_]) {
  initialize(probs);
}

template <typename PROB, typename INDEX>
alias_table_t<PROB, INDEX>::alias_table_t(INDEX length)
  : length_(length), probs_(new block_t[length_]) { }

template <typename PROB, typename INDEX>
alias_table_t<PROB, INDEX>::alias_table_t(alias_table_t&& other)
  : length_(other.length_), probs_(std::move(other.probs_)) { }

template <typename PROB, typename INDEX>
template <typename T>
void alias_table_t<PROB, INDEX>::initialize(T* probs) {
  using namespace aliastable_detail;

  PROB prob_sum = PROB();

  INDEX small_count = 0;
  INDEX large_count = 0;
  std::unique_ptr<PROB[]>  norm_probs(new PROB[length_]);
  std::unique_ptr<INDEX[]> large(new INDEX[length_]);
  std::unique_ptr<INDEX[]> small(new INDEX[length_]);

  for (INDEX i = 0; i < length_; ++i) {
    prob_sum += prob_traits<T, PROB>(probs[i]);
  }

  for (INDEX i = 0; i < length_; ++i) {
    norm_probs[i] = prob_traits<T, PROB>(probs[i]) * length_ / prob_sum;
  }

  for (INDEX i = 0; i < length_; ++i) {
    if (norm_probs[i] < 1.0) {
      small[small_count++] = i;
    } else {
      large[large_count++] = i;
    }
  }

  while (small_count && large_count) {
    INDEX small_idx = small[--small_count];
    INDEX large_idx = large[--large_count];

    probs_[small_idx].prob_  = norm_probs[small_idx];
    probs_[small_idx].alias_ = large_idx;

    norm_probs[large_idx] = norm_probs[large_idx] - (1.0 - norm_probs[small_idx]);

    if (norm_probs[large_idx] < 1.0) {
      small[small_count++] = large_idx;
    } else {
      large[large_count++] = large_idx;
    }
  }

  while (large_count) {
    INDEX idx = large[--large_count];
    probs_[idx].prob_  = 1.0;
    probs_[idx].alias_ = idx;
  }

  while (small_count) {  // can this happen ??
    INDEX idx = small[--small_count];
    probs_[idx].prob_  = 1.0;
    probs_[idx].alias_ = idx;
  }

#ifdef __ALIASTABLE_DEBUG__
  for (INDEX i = 0; i < length_; ++i) {
    LOG(INFO) << "prob: " << probs_[i].prob_ << ", alias: " << probs_[i].alias_;
  }
#endif
}

template <typename PROB, typename INDEX>
void alias_table_t<PROB, INDEX>::initialize(void) {
  initialize(probs_.get());
}

template <typename PROB, typename INDEX>
typename alias_table_t<PROB, INDEX>::block_t& alias_table_t<PROB, INDEX>::operator[] (INDEX i) {
  return probs_[i];
}

template <typename PROB, typename INDEX>
INDEX alias_table_t<PROB, INDEX>::size(void) {
  return length_;
}

template <typename PROB, typename INDEX>
void alias_table_t<PROB, INDEX>::resize(size_t n) {
  std::unique_ptr<block_t[]> new_probs(new block_t[n]);
  memcpy(new_probs.get(), probs_.get(), std::min(n, (size_t)length_) * sizeof(new_probs[0]));

  length_ = n;
  std::swap(new_probs, probs_);
}

template <typename PROB, typename INDEX>
template <typename URNG>
INDEX alias_table_t<PROB, INDEX>::sample(URNG& g) {
  // tiny construction cost
  std::uniform_real_distribution<PROB> dist1(0, 1.0);
  std::uniform_int_distribution<INDEX> dist2(0, length_ - 1);

  INDEX k = dist2(g);
  return dist1(g) < probs_[k].prob_ ? k : probs_[k].alias_;
}

// ************************************************************************************ //

}  // namespace plato

#endif

