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

#include <list>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "glog/logging.h"

namespace plato {

class background_executor : public std::enable_shared_from_this<background_executor> {
  std::list<std::function<void()>> items_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool flushing_ = false;
  std::list<std::thread> threads_;
  size_t max_threads_;
public:
  background_executor(const background_executor&) = delete;
  background_executor& operator=(const background_executor&) = delete;
  background_executor(background_executor&&) = delete;
  background_executor& operator=(background_executor&&) = delete;

  explicit background_executor(size_t max_threads = omp_get_num_threads()) :
    max_threads_(std::max(max_threads, size_t(1))) {};

  ~background_executor() { flush(); }

  void submit(std::function<void()> func) {
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK(!flushing_);

    if (threads_.empty() || (!items_.empty() && threads_.size() < max_threads_)) {
      threads_.emplace_back([this] {
        while(true) {
          while(true) {
            std::function<void()> func;
            {
              std::unique_lock<std::mutex> lock(mutex_);
              if (items_.empty()) {
                break;
              }
              func = std::move(items_.front());
              items_.pop_front();
            }
            func();
          }

          {
            std::unique_lock<std::mutex> lock(mutex_);
            if (flushing_ && items_.empty()) break;
            if (!flushing_ && items_.empty()) cv_.wait(lock);
          }
        }
      });
    }

    items_.emplace_back(std::move(func));
    cv_.notify_one();
  }

  void flush() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      CHECK(!flushing_);
      flushing_ = true;
      cv_.notify_all();
    }

    for (auto& thread : threads_) {
      thread.join();
    }
    threads_.clear();

    {
      std::unique_lock<std::mutex> lock(mutex_);
      CHECK(flushing_);
      flushing_ = false;
    }
  }
};

}