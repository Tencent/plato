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

#ifndef __PLATO_SPINLOCK_HPP__
#define __PLATO_SPINLOCK_HPP__

#include <cstdint>
#include <atomic>

namespace plato {

struct spinlock_t {  // pod-type
  spinlock_t(void): lock_(0) { }

  void lock(void) {
    while (__sync_lock_test_and_set(&lock_, 1)) while (lock_);
  }

  bool try_lock(void) {
    return (1 != __sync_lock_test_and_set(&lock_, 1));
  }

  void unlock(void) {
    __sync_lock_release(&lock_);
  }

protected:
  volatile uint32_t lock_;
} __attribute__((aligned(64)));

struct spinlock_noaligned_t {
  spinlock_noaligned_t(void): lock_(ATOMIC_FLAG_INIT) { }
  spinlock_noaligned_t(const spinlock_noaligned_t&): lock_(ATOMIC_FLAG_INIT) { }

  void lock(void) {
    while (std::atomic_flag_test_and_set_explicit(&lock_, std::memory_order_acquire)) {
      __asm volatile ("pause" ::: "memory");
    }
  }

  void unlock(void) {
    std::atomic_flag_clear_explicit(&lock_, std::memory_order_release);
  }

protected:
  std::atomic_flag lock_;
};

}  // namespace plato

#endif

