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

#ifndef __PLATO_UTIL_LIBSVM_H__
#define __PLATO_UTIL_LIBSVM_H__

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <string>

#include "glog/logging.h"

namespace plato {

using svm_feature_t = float;
using svm_index_t   = uint32_t;
using svm_label_t   = int32_t;

struct svm_node_t {
  svm_index_t   index_;
  svm_feature_t value_;
};

struct svm_sample_t {
  svm_label_t             label_;
  std::vector<svm_node_t> values_;

  template<typename Ar>
  void serialize(Ar &ar) {
    ar & label_ & values_;
  }
};

struct svm_dense_sample_t {
  svm_label_t                label_;
  std::vector<svm_feature_t> values_;

  template<typename Ar>
  void serialize(Ar &ar) {
    ar & label_ & values_;
  }
};

inline bool libsvm_decoder(svm_sample_t* sample, char* input) {
  char* save  = nullptr;
  char* token = nullptr;

  sample->label_ = (svm_label_t)-1;
  sample->values_.clear();

  token = strtok_r(input, " \t", &save);
  if (nullptr == token) {
    LOG(ERROR) << "broken libsvm record: " << input;
    return false;
  }
  sample->label_ = (svm_label_t)strtoll(token, nullptr, 10);

  while (token) {
    // index
    token = strtok_r(nullptr, ":", &save);
    if (nullptr == token) { break; }
    svm_index_t index = (svm_index_t)strtoul(token, nullptr, 10);

    // feature
    token = strtok_r(nullptr, " \t", &save);
    if (nullptr == token) {
      LOG(ERROR) << "broken libsvm record: " << input;
      return false;
    }

    sample->values_.emplace_back(svm_node_t { index, (svm_feature_t)strtod(token, nullptr) });
  }

  return true;
}

struct libsvm_dense_decoder_t {
  svm_index_t max_index_;

  bool operator() (svm_dense_sample_t* dense_sample, char* input) {
    CHECK(0 != max_index_) << "invalid max_index_";

    svm_sample_t sample;
    if (false == libsvm_decoder(&sample, input)) {
      return false;
    }

    dense_sample->label_ = sample.label_;
    dense_sample->values_.clear();
    dense_sample->values_.resize(max_index_, (svm_feature_t)0.0);

    for (auto v: sample.values_) {
      dense_sample->values_[v.index_] = v.value_;
    }

    return true;
  }
};

}

#endif

