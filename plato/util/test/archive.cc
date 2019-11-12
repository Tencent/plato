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

#include "plato/util/stream.hpp"
#include "plato/util/archive.hpp"

#include <memory>

#include "gtest/gtest.h"

using namespace plato;

struct nontrivial_t {
  nontrivial_t(void)
    : a_(0.0), b_(0) { }

  nontrivial_t(double a, uint64_t b)
    : a_(a), b_(b) { }

  template<typename Ar>
  void serialize(Ar &ar) {
      ar & a_ & b_;
  }

  double   a_;
  uint64_t b_;
};

struct trivial_t {
  double   a_;
  uint64_t b_;
} __attribute__((packed));

/***************************************************************************/

TEST(Archive, NonTrivialCreate) {
  oarchive_t<nontrivial_t, mem_ostream_t> oarchive;
  ASSERT_FALSE(oarchive.is_trivial());

  auto buffer = oarchive.get_intrusive_buffer();
  iarchive_t<nontrivial_t, mem_istream_t> iarchive(buffer.data_, buffer.size_, oarchive.count());
  ASSERT_FALSE(iarchive.is_trivial());
}

TEST(Archive, NonTrivialWriteRead) {
  oarchive_t<nontrivial_t, mem_ostream_t> oarchive;
  for (int i = 0; i < 128; ++i) {
    oarchive.emit(nontrivial_t((double)i, (uint64_t)i));
  }

  auto buffer = oarchive.get_intrusive_buffer();
  iarchive_t<nontrivial_t, mem_istream_t> iarchive(buffer.data_, buffer.size_, oarchive.count());
  ASSERT_EQ(128, oarchive.count());

  for (int i = 0; i < 128; ++i) {
    auto msg = iarchive.absorb();

    ASSERT_NE(nullptr, msg);
    ASSERT_FLOAT_EQ((double)i, msg->a_);
    ASSERT_EQ((uint64_t)i, msg->b_);
  }
  ASSERT_EQ(0, iarchive.count());
}

TEST(Archive, NonTrivialCreateWithExistedStream) {
  size_t count = 256;
  mem_ostream_t os;

  os.write(&count, sizeof(count));
  oarchive_t<nontrivial_t, mem_ostream_t> oarchive(std::move(os));
  for (int i = 0; i < (int)count; ++i) {
    oarchive.emit(nontrivial_t((double)i, (uint64_t)i));
  }

  auto buffer = oarchive.get_intrusive_buffer();
  mem_istream_t is(buffer.data_, buffer.size_);
  ASSERT_EQ(sizeof(count), is.read(&count, sizeof(count)));
  ASSERT_EQ(count, 256);

  iarchive_t<nontrivial_t, mem_istream_t> iarchive(std::move(is), count);
  for (int i = 0; i < 256; ++i) {
    auto msg = iarchive.absorb();

    ASSERT_NE(nullptr, msg);
    ASSERT_FLOAT_EQ((double)i, msg->a_);
    ASSERT_EQ((uint64_t)i, msg->b_);
  }
  ASSERT_EQ(0, iarchive.count());
}

TEST(Archive, NonTrivialCreateWithSharedStream) {
  size_t count = 256;
  std::shared_ptr<mem_ostream_t> pos(new mem_ostream_t());

  pos->write(&count, sizeof(count));
  oarchive_t<nontrivial_t, mem_ostream_t> oarchive(pos);
  for (int i = 0; i < (int)count; ++i) {
    oarchive.emit(nontrivial_t((double)i, (uint64_t)i));
  }

  auto buffer = oarchive.get_intrusive_buffer();
  mem_istream_t is(buffer.data_, buffer.size_);
  ASSERT_EQ(sizeof(count), is.read(&count, sizeof(count)));
  ASSERT_EQ(count, 256);

  iarchive_t<nontrivial_t, mem_istream_t> iarchive(std::move(is), count);
  for (int i = 0; i < 256; ++i) {
    auto msg = iarchive.absorb();

    ASSERT_NE(nullptr, msg);
    ASSERT_FLOAT_EQ((double)i, msg->a_);
    ASSERT_EQ((uint64_t)i, msg->b_);
  }
  ASSERT_EQ(0, iarchive.count());
}

/***************************************************************************/

TEST(Archive, TrivialCreate) {
  oarchive_t<trivial_t, mem_ostream_t> oarchive;
  ASSERT_TRUE(oarchive.is_trivial());

  auto buffer = oarchive.get_intrusive_buffer();
  iarchive_t<trivial_t, mem_istream_t> iarchive(buffer.data_, buffer.size_, oarchive.count());
  ASSERT_TRUE(iarchive.is_trivial());
}

TEST(Archive, TrivialWriteRead) {
  oarchive_t<trivial_t, mem_ostream_t> oarchive;
  for (int i = 0; i < 128; ++i) {
    oarchive.emit(trivial_t{(double)i, (uint64_t)i});
  }

  auto buffer = oarchive.get_intrusive_buffer();
  iarchive_t<trivial_t, mem_istream_t> iarchive(buffer.data_, buffer.size_, oarchive.count());
  ASSERT_EQ(128, oarchive.count());

  for (int i = 0; i < 128; ++i) {
    auto msg = iarchive.absorb();

    ASSERT_NE(nullptr, msg);
    ASSERT_FLOAT_EQ((double)i, msg->a_);
    ASSERT_EQ((uint64_t)i, msg->b_);
  }
  ASSERT_EQ(0, iarchive.count());
}

TEST(Archive, TrivialCreateWithExistedStream) {
  size_t count = 256;
  mem_ostream_t os;

  os.write(&count, sizeof(count));
  oarchive_t<trivial_t, mem_ostream_t> oarchive(std::move(os));
  for (int i = 0; i < (int)count; ++i) {
    oarchive.emit(trivial_t{(double)i, (uint64_t)i});
  }

  auto buffer = oarchive.get_intrusive_buffer();
  mem_istream_t is(buffer.data_, buffer.size_);
  ASSERT_EQ(sizeof(count), is.read(&count, sizeof(count)));
  ASSERT_EQ(count, 256);

  iarchive_t<trivial_t, mem_istream_t> iarchive(std::move(is), count);
  for (int i = 0; i < 256; ++i) {
    auto msg = iarchive.absorb();

    ASSERT_NE(nullptr, msg);
    ASSERT_FLOAT_EQ((double)i, msg->a_);
    ASSERT_EQ((uint64_t)i, msg->b_);
  }
  ASSERT_EQ(0, iarchive.count());
}

TEST(Archive, TrivialCreateWithSharedStream) {
  size_t count = 256;
  std::shared_ptr<mem_ostream_t> pos(new mem_ostream_t());

  pos->write(&count, sizeof(count));
  oarchive_t<trivial_t, mem_ostream_t> oarchive(pos);
  for (int i = 0; i < (int)count; ++i) {
    oarchive.emit(trivial_t{(double)i, (uint64_t)i});
  }

  auto buffer = oarchive.get_intrusive_buffer();
  mem_istream_t is(buffer.data_, buffer.size_);
  ASSERT_EQ(sizeof(count), is.read(&count, sizeof(count)));
  ASSERT_EQ(count, 256);

  iarchive_t<trivial_t, mem_istream_t> iarchive(std::move(is), count);
  for (int i = 0; i < 256; ++i) {
    auto msg = iarchive.absorb();

    ASSERT_NE(nullptr, msg);
    ASSERT_FLOAT_EQ((double)i, msg->a_);
    ASSERT_EQ((uint64_t)i, msg->b_);
  }
  ASSERT_EQ(0, iarchive.count());
}

/***************************************************************************/

