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

#include <thread>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "plato/util/thread_local_object.h"

TEST(ThreadObject, ThreadObjectObjectNum) {
  ASSERT_EQ(plato::thread_local_object_detail::objects_num(), 0);
  plato::thread_local_object<int> obj1;
  ASSERT_EQ(plato::thread_local_object_detail::objects_num(), 1);
  {
    plato::thread_local_object<int> obj2;
    ASSERT_EQ(plato::thread_local_object_detail::objects_num(), 2);
  }
  ASSERT_EQ(plato::thread_local_object_detail::objects_num(), 1);
}

TEST(ThreadObject, ThreadObjectThreadNum) {
  plato::thread_local_object<int> obj;
  ASSERT_EQ(obj.objects_num(), 0);
  obj.local();
  ASSERT_EQ(obj.objects_num(), 1);
  ASSERT_EQ(*obj.local(), 0);
  (*obj.local())++;
  ASSERT_EQ(*obj.local(), 1);
  ASSERT_EQ(obj.objects_num(), 1);

  std::thread t([&] {
    ASSERT_EQ(obj.objects_num(), 1);
    ASSERT_EQ(*obj.local(), 0);
    ASSERT_EQ(obj.objects_num(), 2);
    (*obj.local())++;
    ASSERT_EQ(*obj.local(), 1);
  });
  t.join();
  ASSERT_EQ(*obj.local(), 1);
  ASSERT_EQ(obj.objects_num(), 2);
}

TEST(ThreadObject, ThreadObjectCounter) {
  plato::thread_local_counter counter;

  int local_num = 100000;

  auto func = [&] {
    ASSERT_EQ(counter.local(), 0);
    for (int i = 0; i < local_num; i++) {
      counter.local()++;
    }
    ASSERT_EQ(counter.local(), local_num);
  };

  std::thread t1(func);
  std::thread t2(func);

  t1.join();
  t2.join();

  ASSERT_EQ(counter.reduce_sum(), local_num * 2);
}

TEST(ThreadObject, ThreadObjectCounterMultiThread) {
  plato::thread_local_counter counter;

  int local_num = 1000;
  int iter_num = 100;

  std::vector<std::thread> threads;
  for (int i = 0; i < iter_num; ++i) {
    threads.emplace_back([&] {
      ASSERT_EQ(counter.local(), 0);
      for (int c = 0; c < local_num; c++) {
        counter.local()++;
      }
      ASSERT_EQ(counter.local(), local_num);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  ASSERT_EQ(counter.reduce_sum(), local_num * iter_num);
}