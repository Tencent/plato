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

#include <future>

#include "thread_local_object.h"
#include "defer.hpp"

namespace plato {

namespace thread_local_object_detail {

struct dist_obj;
struct local_obj;

static constexpr int objs_capacity = 4096;
static std::mutex g_mutex_;

static std::vector<std::shared_ptr<dist_obj>> dist_objs_(objs_capacity);
static std::vector<std::mutex> mutex_v_(objs_capacity);
static thread_local std::vector<local_obj> local_objs_(objs_capacity);

struct local_obj {
  std::shared_ptr<void> local_obj_p_;
  std::shared_ptr<dist_obj> dist_obj_;
  boost::intrusive::list_member_hook<> link_;
  ~local_obj();
};

struct dist_obj : public std::enable_shared_from_this<dist_obj> {
  using objs_list_t = boost::intrusive::list<local_obj, boost::intrusive::member_hook<local_obj, boost::intrusive::list_member_hook<>, &local_obj::link_>>;

  std::function<void* ()> construction_;
  std::function<void(void *)> destruction_;

  bool closing_ = false;
  objs_list_t objs_list_;
  std::list<std::shared_ptr<void>> objs_p_list_;

  std::promise<void> pro_;
  std::future<void> fut_ = pro_.get_future();

  dist_obj(std::function<void* ()> construction, std::function<void(void *)> destruction) :
    construction_(std::move(construction)), destruction_(std::move(destruction)) { }

  ~dist_obj() {
    objs_p_list_.clear();
    pro_.set_value();
  }
};

local_obj::~local_obj() {
  int id = this - local_objs_.data();
  std::lock_guard<std::mutex> lock(mutex_v_[id]);
  if (link_.is_linked()) {
    CHECK(dist_obj_ && local_obj_p_);
    local_obj_p_.reset();
    dist_obj_->objs_list_.erase(dist_obj_->objs_list_.iterator_to(*this));
    dist_obj_.reset();
  }
}

int create_object(std::function<void*()> construction, std::function<void(void *)> destruction) {
  std::lock_guard<std::mutex> lock(g_mutex_);
  for (int id = 0; id < objs_capacity; ++id) {
    if (!dist_objs_[id]) {
      dist_objs_[id] = std::make_shared<dist_obj>(std::move(construction), std::move(destruction));
      return id;
    }
  }
  return -1;
}

void delete_object(int id) {
  if (id < 0 || id >= objs_capacity) throw std::runtime_error("invalid object id");
  std::shared_ptr<dist_obj> dist_obj_ = dist_objs_[id];
  if (!dist_obj_) throw std::runtime_error("object not exist.");

  {
    std::lock_guard<std::mutex> lock(mutex_v_[id]);
    dist_obj_->closing_ = true;

    while (!dist_obj_->objs_list_.empty()) {
      local_obj& obj = dist_obj_->objs_list_.front();
      obj.local_obj_p_.reset();
      obj.dist_obj_.reset();
      dist_obj_->objs_list_.pop_front();
    }
    dist_objs_[id].reset();
  }

  std::future<void> fut = std::move(dist_obj_->fut_);
  dist_obj_.reset();
  fut.get();
}

void* get_local_object(int id) {
  static thread_local local_obj* local_objs_p_;
  if (__glibc_unlikely(!local_objs_p_)) {
    // local_objs_ has a non-trivial constructor which means that the compiler
    // must make sure the instance local to the current thread has been
    // constructed before each access.
    // Unfortunately, this means that GCC will emit an unconditional call
    // to __tls_init(), which may incurr a noticeable overhead.
    // This can be solved by adding an easily predictable branch checking
    // whether the object has already been constructed.
    local_objs_p_ = local_objs_.data();
  }

  local_obj& obj = local_objs_p_[id];
  if (__glibc_unlikely(!obj.local_obj_p_.get())) {
    std::shared_ptr<dist_obj> dist_obj_ = dist_objs_[id];
    if (!dist_obj_) throw std::runtime_error("object not exist.");
    std::shared_ptr<void> p(dist_obj_->construction_(), dist_obj_->destruction_);
    {
      std::lock_guard<std::mutex> lock(mutex_v_[id]);
      if (dist_obj_->closing_) throw std::runtime_error("dist_obj is closing.");
      CHECK(!obj.link_.is_linked() && !obj.local_obj_p_ && !obj.dist_obj_);
      dist_obj_->objs_list_.push_back(obj);
      dist_obj_->objs_p_list_.push_back(p);
      obj.local_obj_p_ = p;
      obj.dist_obj_ = dist_obj_;
    }
  }

  return obj.local_obj_p_.get();
}

unsigned objects_num() {
  std::lock_guard<std::mutex> lock(g_mutex_);
  size_t num = 0;
  for (std::shared_ptr<dist_obj> dist_obs_ : dist_objs_) {
    if (dist_obs_) {
      num++;
    }
  }
  return num;
}

unsigned objects_num(int id) {
  if (id < 0 || id >= objs_capacity) throw std::runtime_error("invalid object id");
  std::shared_ptr<dist_obj> dist_obs_ = dist_objs_[id];
  std::lock_guard<std::mutex> lock(mutex_v_[id]);
  return dist_obs_->objs_p_list_.size();
}

void object_foreach(int id, std::function<void(void*)> reducer) {
  if (id < 0 || id >= objs_capacity) throw std::runtime_error("invalid object id");
  std::shared_ptr<dist_obj> dist_obj_ = dist_objs_[id];
  if (!dist_obj_) throw std::runtime_error("object not exist.");

  std::lock_guard<std::mutex> lock(mutex_v_[id]);
  for (std::shared_ptr<void>& p : dist_obj_->objs_p_list_) {
    reducer(p.get());
  }
}

}

}
