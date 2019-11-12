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

#include "plato/graph/parsers.hpp"

#include <fstream>

#include "gtest/gtest.h"

TEST(Serialization, ParseFromCSVWithoutDecoder) { // push state machine
    std::fstream fi("data/graph/raw_graph_10_9.csv", std::fstream::in);

    using EData = plato::empty_t;
    using EUnit = plato::edge_unit_t<EData, plato::vid_t>;

    size_t count = 0;
    EUnit blocks[1024];

    auto blockcallback = [&](EUnit* pInput, size_t size) {
        memcpy(&blocks[count], pInput, sizeof(EUnit) * size);
        count += size;
        return true;
    };

    ssize_t rc = plato::csv_parser<std::fstream, EData, plato::vid_t>(fi, blockcallback, plato::dummy_decoder<EData>);
    ASSERT_EQ(9, rc);

    ASSERT_EQ(0, blocks[0].src_); ASSERT_EQ(1, blocks[0].dst_);
    ASSERT_EQ(1, blocks[1].src_); ASSERT_EQ(2, blocks[1].dst_);
    ASSERT_EQ(2, blocks[2].src_); ASSERT_EQ(3, blocks[2].dst_);
    ASSERT_EQ(3, blocks[3].src_); ASSERT_EQ(4, blocks[3].dst_);
    ASSERT_EQ(4, blocks[4].src_); ASSERT_EQ(5, blocks[4].dst_);
    ASSERT_EQ(5, blocks[5].src_); ASSERT_EQ(6, blocks[5].dst_);
    ASSERT_EQ(6, blocks[6].src_); ASSERT_EQ(7, blocks[6].dst_);
    ASSERT_EQ(7, blocks[7].src_); ASSERT_EQ(8, blocks[7].dst_);
    ASSERT_EQ(8, blocks[8].src_); ASSERT_EQ(9, blocks[8].dst_);
}

TEST(Serialization, ParseFromCSVWithDecoder) { // push state machine
    std::fstream fi("data/graph/graph_10_9.csv", std::fstream::in);

    using EData = float;
    using EUnit = plato::edge_unit_t<EData, plato::vid_t>;

    size_t count = 0;
    EUnit blocks[1024];

    auto blockcallback = [&](EUnit* pInput, size_t size) {
        memcpy(&blocks[count], pInput, sizeof(EUnit) * size);
        count += size;
        return true;
    };

    ssize_t rc = plato::csv_parser<std::fstream, EData, plato::vid_t>(fi, blockcallback, plato::float_decoder);
    ASSERT_EQ(9, rc);

    ASSERT_EQ(0, blocks[0].src_); ASSERT_EQ(1, blocks[0].dst_);
    ASSERT_EQ(1, blocks[1].src_); ASSERT_EQ(2, blocks[1].dst_);
    ASSERT_EQ(2, blocks[2].src_); ASSERT_EQ(3, blocks[2].dst_);
    ASSERT_EQ(3, blocks[3].src_); ASSERT_EQ(4, blocks[3].dst_);
    ASSERT_EQ(4, blocks[4].src_); ASSERT_EQ(5, blocks[4].dst_);
    ASSERT_EQ(5, blocks[5].src_); ASSERT_EQ(6, blocks[5].dst_);
    ASSERT_EQ(6, blocks[6].src_); ASSERT_EQ(7, blocks[6].dst_);
    ASSERT_EQ(7, blocks[7].src_); ASSERT_EQ(8, blocks[7].dst_);
    ASSERT_EQ(8, blocks[8].src_); ASSERT_EQ(9, blocks[8].dst_);

    ASSERT_FLOAT_EQ(0.123, blocks[0].edata_);
    ASSERT_FLOAT_EQ(1.779, blocks[1].edata_);
    ASSERT_FLOAT_EQ(2.451, blocks[2].edata_);
    ASSERT_FLOAT_EQ(19.22, blocks[3].edata_);
    ASSERT_FLOAT_EQ(0.334, blocks[4].edata_);
    ASSERT_FLOAT_EQ(102.1, blocks[5].edata_);
    ASSERT_FLOAT_EQ(738.5, blocks[6].edata_);
    ASSERT_FLOAT_EQ(999.9, blocks[7].edata_);
    ASSERT_FLOAT_EQ(1.111, blocks[8].edata_);
}

