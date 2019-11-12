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

#include "plato/util/libsvm.hpp"

#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(LIBSVM, DecodeFromString) {
  char input[255] = "12 0:0.1 1:0.2 100:0.4 3:1.2 10:0.44";

  plato::svm_sample_t sample;
  ASSERT_TRUE(libsvm_decoder(&sample, input));

  ASSERT_EQ(sample.values_.size(), 5);
  ASSERT_EQ(sample.label_, 12);

  ASSERT_EQ(sample.values_[0].index_, 0);   ASSERT_FLOAT_EQ(sample.values_[0].value_, 0.1);
  ASSERT_EQ(sample.values_[1].index_, 1);   ASSERT_FLOAT_EQ(sample.values_[1].value_, 0.2);
  ASSERT_EQ(sample.values_[2].index_, 100); ASSERT_FLOAT_EQ(sample.values_[2].value_, 0.4);
  ASSERT_EQ(sample.values_[3].index_, 3);   ASSERT_FLOAT_EQ(sample.values_[3].value_, 1.2);
  ASSERT_EQ(sample.values_[4].index_, 10);  ASSERT_FLOAT_EQ(sample.values_[4].value_, 0.44);
}

TEST(LIBSVM, DecodeDenseFromString) {
  char input[255] = "12 0:0.1 1:0.2 100:0.4 3:1.2 10:0.44";

  plato::svm_dense_sample_t sample;
  plato::libsvm_dense_decoder_t decoder { 101 };

  ASSERT_TRUE(decoder(&sample, input));

  ASSERT_EQ(sample.values_.size(), 101);
  ASSERT_EQ(sample.label_, 12);

  for (uint32_t i = 0; i < 101; ++i) {
    if (0   == i) { ASSERT_FLOAT_EQ(sample.values_[i], 0.1);  continue; }
    if (1   == i) { ASSERT_FLOAT_EQ(sample.values_[i], 0.2);  continue; }
    if (100 == i) { ASSERT_FLOAT_EQ(sample.values_[i], 0.4);  continue; }
    if (3   == i) { ASSERT_FLOAT_EQ(sample.values_[i], 1.2);  continue; }
    if (10  == i) { ASSERT_FLOAT_EQ(sample.values_[i], 0.44); continue; }

    ASSERT_FLOAT_EQ(sample.values_[i], 0.0);
  }
}

