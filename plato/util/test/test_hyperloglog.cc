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

#include <cmath>
#include <cfloat>
#include "gtest/gtest.h"
#include "plato/util/hyperloglog.hpp"
#include "glog/logging.h"

TEST(HyperLogLog, Add) {
  plato::hyperloglog_t<6> hll;
  hll.init();
  hll.clear();

  for (int i = 0; i < 1000; ++i) {
    hll.add(&i, sizeof(i));
  }

  double cardinality = hll.estimate();
  ASSERT_GE(0.25, fabs(cardinality - 1000) / 1000);
}

TEST(HyperLogLog, Merge) {
  plato::hyperloglog_t<12> hll1;
  hll1.init();
  hll1.clear();
  plato::hyperloglog_t<12> hll2;
  hll2.init();
  hll2.clear();
;
  for (int i = 0; i < 1000; ++i) {
    hll1.add(&i, sizeof(i));
    hll2.add(&i, sizeof(i));
  }
  double cardinality1 = hll1.estimate();
  hll1.merge(hll2);
  double cardinality2 = hll1.estimate();
  ASSERT_GE(DBL_EPSILON, fabs(cardinality1 - cardinality2));

  hll2.clear();
  int x = 1000;
  hll2.add(&x, sizeof(x));
  hll1.merge(hll2);
  cardinality2 = hll1.estimate();
  ASSERT_GE(cardinality2, cardinality1);
}
