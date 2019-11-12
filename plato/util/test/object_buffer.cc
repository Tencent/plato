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

#include "plato/util/object_buffer.hpp"

#include <cstdint>
#include <mutex>
#include <vector>

#include "omp.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

struct nontrivial_t {
  nontrivial_t(void)
    : a_(0), b_(0) { }

  template <typename T>
  nontrivial_t(T v) : a_(v), b_(v) { }

  nontrivial_t(uint32_t a, uint64_t b)
    : a_(a), b_(b) { }

  bool operator== (const nontrivial_t& other) const {
    return a_ == other.a_ && b_ == other.b_;
  }

  bool operator< (const nontrivial_t& other) const {
    return b_ < other.b_;
  }

  template<typename Ar>
  void serialize(Ar &ar) {
    ar & a_ & b_;
  }

  uint32_t a_;
  uint64_t b_;
};

struct trivial_t {
  double   a_;
  uint64_t b_;

  trivial_t() = default;

  template <typename T>
  trivial_t(T v) : a_(v), b_(v) { }

  template <typename T1, typename T2>
  trivial_t(T1 a, T2 b) : a_(a), b_(b) { }

  friend inline bool operator<(const trivial_t l, const trivial_t r) noexcept {
    return l.b_ < r.b_;
  }
  friend inline bool operator== (const trivial_t &l, const trivial_t &r) {
    return l.a_ == r.a_ && l.b_ == r.b_;
  }
} __attribute__((packed));

TEST(Buffer, ObjectBufferTrivialEmplaceBack) {
  size_t capacity = 514;

  std::vector<uint32_t> elements;
  plato::object_buffer_t<uint32_t> objs(capacity);

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.emplace_back(i);
  }

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  ASSERT_EQ(elements.size(), objs.size());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_THAT(elements, testing::Contains(objs[i]));
  }
}

TEST(Buffer, ObjectBufferNonTrivialEmplaceBack) {
  size_t capacity = 514;

  std::vector<nontrivial_t> elements;
  plato::object_buffer_t<nontrivial_t> objs(capacity);

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.emplace_back(i);
  }

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  ASSERT_EQ(elements.size(), objs.size());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_THAT(elements, testing::Contains(objs[i]));
  }
}

TEST(Buffer, ObjectBufferTrivialPushBack) {
  size_t capacity = 514;

  std::vector<uint32_t> elements;
  plato::object_buffer_t<uint32_t> objs(capacity);

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.push_back(elements[i]);
  }

  ASSERT_EQ(elements.size(), objs.size());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_THAT(elements, testing::Contains(objs[i]));
  }
}

TEST(Buffer, ObjectBufferNonTrivialPushBack) {
  size_t capacity = 514;

  std::vector<nontrivial_t> elements;
  plato::object_buffer_t<nontrivial_t> objs(capacity);

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.push_back(elements[i]);
  }

  ASSERT_EQ(elements.size(), objs.size());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_THAT(elements, testing::Contains(objs[i]));
  }
}

TEST(Buffer, ObjectBufferTrivialPushBackV) {
  size_t capacity = 514;
  size_t amount   = 4;
  size_t chunks   = capacity / amount + 1;

  std::vector<uint32_t> elements;
  plato::object_buffer_t<uint32_t> objs(capacity);

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < chunks; ++i) {
    size_t start = amount * i;
    size_t chunk = amount;
    if (start + chunk >= capacity) { chunk = capacity - start; }
    objs.push_back(&elements[start], chunk);
  }

  ASSERT_EQ(elements.size(), objs.size());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_THAT(elements, testing::Contains(objs[i]));
  }
}

TEST(Buffer, ObjectBufferNonTrivialPushBackV) {
  size_t capacity = 514;
  size_t amount   = 4;
  size_t chunks   = capacity / amount + 1;

  std::vector<nontrivial_t> elements;
  plato::object_buffer_t<nontrivial_t> objs(capacity);

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < chunks; ++i) {
    size_t start = amount * i;
    size_t chunk = amount;
    if (start + chunk >= capacity) { chunk = capacity - start; }
    objs.push_back(&elements[start], chunk);
  }

  ASSERT_EQ(elements.size(), objs.size());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_THAT(elements, testing::Contains(objs[i]));
  }
}

TEST(Buffer, ObjectBufferTrivialTravse) {
  size_t capacity = 514;

  std::vector<uint32_t> elements;
  plato::object_buffer_t<uint32_t> objs(capacity);

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.emplace_back(i);
  }

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  std::mutex mtx;
  std::vector<uint32_t> __elements;

  objs.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 12;
    while (objs.next_chunk([&](size_t, uint32_t* pv) {
      mtx.lock();
      __elements.emplace_back(*pv);
      mtx.unlock();
    }, &chunk_size)) { }
  }

  ASSERT_EQ(__elements.size(), elements.size());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_THAT(elements, testing::Contains(__elements[i]));
  }
}

TEST(Buffer, ObjectFileBufferConstruct) {
  plato::object_file_buffer_t<trivial_t> objs;
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(objs.is_trivial());

  objs.reset_traversal();
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, trivial_t*) {
      ASSERT_TRUE(false);
    },
    &chunk_size
  )) { }
}

TEST(Buffer, ObjectFileBufferTrivialEmpty) {
  size_t capacity = MBYTES;
  plato::object_file_buffer_t<trivial_t> objs(capacity);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(objs.is_trivial());

  objs.reset_traversal();
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, trivial_t*) {
      ASSERT_TRUE(false);
    },
    &chunk_size
  )) { }
}

TEST(Buffer, ObjectFileBufferTrivialPushBackAligned) {
  size_t capacity = 4 * MBYTES * 2;

  std::vector<trivial_t> elements;
  plato::object_file_buffer_t<trivial_t> objs(capacity);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(objs.is_trivial());

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.push_back(i);
  }

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  ASSERT_EQ(elements.size(), objs.size());

  objs.reset_traversal();
  std::vector<trivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, trivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectFileBufferTrivialPushBack) {
  size_t capacity = 4 * MBYTES * 2.3;

  std::vector<trivial_t> elements;
  plato::object_file_buffer_t<trivial_t> objs(capacity);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(objs.is_trivial());

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.push_back(elements[i]);
  }

  ASSERT_EQ(elements.size(), objs.size());

  objs.reset_traversal();
  std::vector<trivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, trivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectFileBufferTrivialPushBackV) {
  size_t capacity = 4 * MBYTES * 2.3;
  size_t amount   = 4;
  size_t chunks   = capacity / amount + 1;

  std::vector<trivial_t> elements;
  plato::object_file_buffer_t<trivial_t> objs(capacity);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(objs.is_trivial());

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < chunks; ++i) {
    size_t start = amount * i;
    size_t chunk = amount;
    if (start + chunk >= capacity) { chunk = capacity - start; }
    objs.push_back(&elements[start], chunk);
  }

  ASSERT_EQ(elements.size(), objs.size());

  objs.reset_traversal();
  std::vector<trivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, trivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectFileBufferTrivialPushBackBigV) {
  size_t capacity = 4 * MBYTES * 2.3;

  std::vector<trivial_t> elements;
  plato::object_file_buffer_t<trivial_t> objs(capacity);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(objs.is_trivial());

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  objs.push_back(elements.data(), capacity);

  ASSERT_EQ(elements.size(), objs.size());

  objs.reset_traversal();
  std::vector<trivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, trivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectFileBufferTrivialTravse) {
  size_t capacity = 4 * MBYTES * 2.3;

  std::vector<trivial_t> elements;
  plato::object_file_buffer_t<trivial_t> objs(capacity);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(objs.is_trivial());

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.push_back(i);
  }

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  std::mutex mtx;
  std::vector<trivial_t> __elements;

  objs.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 12;
    while (objs.next_chunk([&](size_t, trivial_t* pv) {
      mtx.lock();
      __elements.emplace_back(*pv);
      mtx.unlock();
    }, &chunk_size)) { }
  }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectFileBufferNonTrivialEmpty) {
  size_t capacity = MBYTES;
  plato::object_file_buffer_t<nontrivial_t> objs(capacity);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(!objs.is_trivial());

  objs.reset_traversal();
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, nontrivial_t*) {
      ASSERT_TRUE(false);
    },
    &chunk_size
  )) { }
}

TEST(Buffer, ObjectFileBufferNonTrivialPushBackAligned) {
  size_t capacity = 4 * MBYTES * 2;
  size_t mem_size = capacity * sizeof(nontrivial_t);

  std::vector<nontrivial_t> elements;
  plato::object_file_buffer_t<nontrivial_t> objs(mem_size);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(!objs.is_trivial());

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.push_back(i);
  }

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  objs.reset_traversal();
  std::vector<nontrivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, nontrivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectFileBufferNonTrivialPushBack) {
  size_t capacity = 4 * MBYTES * 2.3;
  size_t mem_size = capacity * sizeof(nontrivial_t);

  std::vector<nontrivial_t> elements;
  plato::object_file_buffer_t<nontrivial_t> objs(mem_size);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(!objs.is_trivial());

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.push_back(elements[i]);
  }

  objs.reset_traversal();
  std::vector<nontrivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, nontrivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectFileBufferNonTrivialPushBackV) {
  size_t capacity = 4 * MBYTES * 2.3;
  size_t mem_size = capacity * sizeof(nontrivial_t);
  size_t amount   = 4;
  size_t chunks   = capacity / amount + 1;

  std::vector<nontrivial_t> elements;
  plato::object_file_buffer_t<nontrivial_t> objs(mem_size);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(!objs.is_trivial());

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < chunks; ++i) {
    size_t start = amount * i;
    size_t chunk = amount;
    if (start + chunk >= capacity) { chunk = capacity - start; }
    objs.push_back(&elements[start], chunk);
  }

  objs.reset_traversal();
  std::vector<nontrivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, nontrivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectFileBufferNonTrivialPushBackBigV) {
  size_t capacity = 4 * MBYTES * 2.3;
  size_t mem_size = capacity * sizeof(nontrivial_t);

  std::vector<nontrivial_t> elements;
  plato::object_file_buffer_t<nontrivial_t> objs(mem_size);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(!objs.is_trivial());

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  objs.push_back(elements.data(), capacity);

  objs.reset_traversal();
  std::vector<nontrivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, nontrivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectFileBufferNonTrivialTravse) {
  size_t capacity = 4 * MBYTES * 2.3;
  size_t mem_size = capacity * sizeof(nontrivial_t);

  std::vector<nontrivial_t> elements;
  plato::object_file_buffer_t<nontrivial_t> objs(mem_size);
  ASSERT_EQ(0, objs.size());
  ASSERT_TRUE(!objs.is_trivial());

  #pragma omp parallel for
  for (size_t i = 0; i < capacity; ++i) {
    objs.push_back(i);
  }

  for (size_t i = 0; i < capacity; ++i) {
    elements.emplace_back(i);
  }

  std::mutex mtx;
  std::vector<nontrivial_t> __elements;

  objs.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 12;
    while (objs.next_chunk([&](size_t, nontrivial_t* pv) {
      mtx.lock();
      __elements.emplace_back(*pv);
      mtx.unlock();
    }, &chunk_size)) { }
  }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < capacity; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectDfsBufferTrivialEmpty) {
  plato::object_dfs_buffer_t<trivial_t> objs("");
  ASSERT_TRUE(objs.is_trivial());

  objs.reset_traversal();
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, trivial_t*) {
      ASSERT_TRUE(false);
    },
    &chunk_size
  )) { }
}

TEST(Buffer, ObjectDfsBufferTrivialPushBack) {
  std::vector<trivial_t> elements;
  plato::object_dfs_buffer_t<trivial_t> objs("");
  ASSERT_TRUE(objs.is_trivial());

  for (size_t i = 0; i < MBYTES; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < MBYTES; ++i) {
    objs.push_back(elements[i]);
  }

  objs.reset_traversal();
  std::vector<trivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, trivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < MBYTES; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectDfsBufferTrivialPushBackV) {
  std::vector<trivial_t> elements;
  plato::object_dfs_buffer_t<trivial_t> objs("");
  ASSERT_TRUE(objs.is_trivial());

  for (size_t i = 0; i < MBYTES; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < MBYTES / 4; ++i) {
    size_t start = 4 * i;
    size_t chunk = 4;
    if (start + chunk >= MBYTES) { chunk = MBYTES - start; }
    objs.push_back(&elements[start], chunk);
  }

  objs.reset_traversal();
  std::vector<trivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, trivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < MBYTES; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectDfsBufferTrivialTravse) {
  std::vector<trivial_t> elements;
  plato::object_dfs_buffer_t<trivial_t> objs("");
  ASSERT_TRUE(objs.is_trivial());

  #pragma omp parallel for
  for (size_t i = 0; i < MBYTES; ++i) {
    objs.push_back(i);
  }

  for (size_t i = 0; i < MBYTES; ++i) {
    elements.emplace_back(i);
  }

  std::mutex mtx;
  std::vector<trivial_t> __elements;

  objs.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 12;
    while (objs.next_chunk([&](size_t, trivial_t* pv) {
      mtx.lock();
      __elements.emplace_back(*pv);
      mtx.unlock();
    }, &chunk_size)) { }
  }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < MBYTES; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectDfsBufferNonTrivialEmpty) {
  plato::object_dfs_buffer_t<nontrivial_t> objs("");
  ASSERT_TRUE(!objs.is_trivial());

  objs.reset_traversal();
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, nontrivial_t*) {
      ASSERT_TRUE(false);
    },
    &chunk_size
  )) { }
}

TEST(Buffer, ObjectDfsBufferNonTrivialPushBack) {
  std::vector<nontrivial_t> elements;
  plato::object_dfs_buffer_t<nontrivial_t> objs("");
  ASSERT_TRUE(!objs.is_trivial());

  for (size_t i = 0; i < MBYTES; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < MBYTES; ++i) {
    objs.push_back(elements[i]);
  }

  objs.reset_traversal();
  std::vector<nontrivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, nontrivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < MBYTES; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectDfsBufferNonTrivialPushBackV) {
  std::vector<nontrivial_t> elements;
  plato::object_dfs_buffer_t<nontrivial_t> objs("");
  ASSERT_TRUE(!objs.is_trivial());

  for (size_t i = 0; i < MBYTES; ++i) {
    elements.emplace_back(i);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < MBYTES / 4; ++i) {
    size_t start = 4 * i;
    size_t chunk = 4;
    if (start + chunk >= MBYTES) { chunk = MBYTES - start; }
    objs.push_back(&elements[start], chunk);
  }

  objs.reset_traversal();
  std::vector<nontrivial_t> __elements;
  size_t chunk_size = 1;
  while (objs.next_chunk(
    [&] (size_t, nontrivial_t* pv) {
      __elements.emplace_back(*pv);
    },
    &chunk_size
  )) { }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < MBYTES; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}

TEST(Buffer, ObjectDfsBufferNonTrivialTravse) {
  std::vector<nontrivial_t> elements;
  plato::object_dfs_buffer_t<nontrivial_t> objs("");
  ASSERT_TRUE(!objs.is_trivial());

  #pragma omp parallel for
  for (size_t i = 0; i < MBYTES; ++i) {
    objs.push_back(i);
  }

  for (size_t i = 0; i < MBYTES; ++i) {
    elements.emplace_back(i);
  }

  std::mutex mtx;
  std::vector<nontrivial_t> __elements;

  objs.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 12;
    while (objs.next_chunk([&](size_t, nontrivial_t* pv) {
      mtx.lock();
      __elements.emplace_back(*pv);
      mtx.unlock();
    }, &chunk_size)) { }
  }

  ASSERT_EQ(__elements.size(), elements.size());
  std::sort(__elements.begin(), __elements.end());
  for (size_t i = 0; i < MBYTES; ++i) {
    ASSERT_EQ(__elements[i], elements[i]);
  }
}
