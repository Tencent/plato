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

#ifndef __PLATO_PARSERS_H__
#define __PLATO_PARSERS_H__

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <type_traits>

#include "glog/logging.h"
#include "boost/format.hpp"
#include "boost/algorithm/string.hpp"

#include "plato/graph/base.hpp"
#include "plato/util/libsvm.hpp"

namespace plato {

/*
 * \brief decoder_t, decode edge data from string
 *
 * \param pOutput   output
 * \param sInput    unresolve c-string, end with '\0', you can modify it
 *
 * \return true -- continue parse stage, false -- abort parse
 *
 **/
template <typename EdgeData>
using decoder_t = std::function<bool(EdgeData*, char*)>;

/**
 * @brief null decoder
 * @tparam EdgeData
 * @return
 */
template <typename EdgeData>
inline bool dummy_decoder(EdgeData* /*pOutput*/, char* /*sInput*/) {
  return true;
}

/**
 * @brief decoder
 * @param output - result
 * @param s_input - input string
 * @return is success
 */
inline bool float_decoder(float* output, char* s_input) {
  if (nullptr == s_input) return false;
  *output = std::strtof(s_input, nullptr);
  return true;
}

/**
 * @brief decoder
 * @param output - result
 * @param s_input - input string
 * @return is success
 */
inline bool double_decoder(double* output, char* s_input) {
  if (nullptr == s_input) return false;
  *output = std::strtod(s_input, nullptr);
  return true;
}

/**
 * @brief decoder
 * @param output - result
 * @param s_input - input string
 * @return is success
 */
inline bool uint8_t_decoder(uint8_t* output, char* s_input) {
  if (nullptr == s_input) return false;
  *output = (uint8_t)std::strtoul(s_input, nullptr, 0);
  return true;
}

/**
 * @brief decoder
 * @param output - result
 * @param s_input - input string
 * @return is success
 */
inline bool uint16_t_decoder(uint16_t* output, char* s_input) {
  if (nullptr == s_input) return false;
  *output = (uint16_t)std::strtoul(s_input, nullptr, 0);
  return true;
}

/**
 * @brief decoder
 * @param output - result
 * @param s_input - input string
 * @return is success
 */
inline bool uint32_t_decoder(uint32_t* output, char* s_input) {
  if (nullptr == s_input) return false;
  *output = (uint32_t)std::strtoul(s_input, nullptr, 0);
  return true;
}

/**
 * @brief decoder
 * @param output - result
 * @param s_input - input string
 * @return is success
 */
inline bool uint64_t_decoder(uint64_t* output, char* s_input) {
  if (nullptr == s_input) return false;
  *output = (uint64_t)strtoull(s_input, nullptr, 0);
  return true;
}

/**
 * @brief decoder
 * @param output - result
 * @param s_input - input string
 * @return is success
 */
inline bool int64_t_decoder(int64_t* output, char* s_input) {
  if (nullptr == s_input) return false;
  *output = strtoull(s_input, nullptr, 0);
  return true;
}

template <typename EdgeData>
class decoder_with_default_t {
public:
  decoder_with_default_t() :
    default_value_(EdgeData {}){
  }
  /**
   * @brief
   * @param default_value
   */
  decoder_with_default_t(EdgeData default_value) :
    default_value_(default_value){
  }

  /**
   * @brief
   * @param output
   * @param s_input
   * @return
   */
  inline bool operator()(EdgeData* output, char* s_input) {
    if (nullptr != s_input) {
      *output = (EdgeData)std::strtod(s_input, nullptr);
    }
    else {
      *output = (EdgeData)default_value_;
    }
    return true;
  }
private:
  EdgeData default_value_;
};

/*
 * \brief edge_parser_t,  parse from input stream, call user provide function when buffer if full
 *                        edge_parser_t must be reentrant
 *
 * \param fin,          input stream
 * \param callback      user provide callback function for take edges away
 *
 * \return:
 *      >=0 edges parse from fin
 *      <0  something wrong happened
 *
 *
 *
 * \brief blockcallback_t, user provide function to deal with extracted edges
 *                  blockcallback_t must be reentrant
 *
 * \param buffer,   edge_unit_t's buffer, input
 * \param lenth,    buffer's size
 *
 * \return true --  continue parse stage, false -- abort parse
 *
 **/
template <typename EdgeData, typename VID_T = vid_t>
using blockcallback_t = std::function<bool(edge_unit_t<EdgeData, VID_T>*, size_t)>;

template <typename STREAM_T, typename EdgeData, typename VID_T = vid_t>
using edge_parser_t = std::function<ssize_t(STREAM_T&, blockcallback_t<EdgeData, VID_T>, decoder_t<EdgeData>)>;

// build-in parsers

/**
 * @brief parser for csv format
 * Comma-Separated Values, rfc4180
 * https://tools.ietf.org/html/rfc4180
 * @tparam STREAM_T
 * @tparam EdgeData
 * @tparam VID_T
 * @param fin
 * @param callback
 * @param decoder
 * @return
 */
template <typename STREAM_T, typename EdgeData, typename VID_T = vid_t>
ssize_t csv_parser(STREAM_T& fin, blockcallback_t<EdgeData, VID_T> callback, decoder_t<EdgeData> decoder) {
  ssize_t total_count = 0;
  size_t  count = 0;
  size_t  valid_splits = std::is_same<EdgeData, empty_t>::value ? 2 : 3;

  char* pSave  = nullptr;
  char* pToken = nullptr;
  char* pLog   = nullptr;
  std::unique_ptr<char[]> sInput(new char[HUGESIZE]);
  std::unique_ptr< edge_unit_t<EdgeData, VID_T>[] > buffer(new edge_unit_t<EdgeData, VID_T>[HUGESIZE]);

  while (fin.good() && (false == fin.eof())) {
    fin.getline(sInput.get(), HUGESIZE);
    if (fin.fail() || ('\0' == sInput[0])) {
      continue;
    }

    pLog   = sInput.get();
    pToken = strtok_r(sInput.get(), ",", &pSave);
    if (nullptr == pToken) {
      LOG(WARNING) << boost::format("can not extract source from (%s)") % pLog;
      continue;
    }

    auto src = strtoul(pToken, nullptr, 10);
    CHECK(src <= std::numeric_limits<VID_T>::max()) << "src: " << src << " exceed max value";
    buffer[count].src_ = src;

    pLog = pToken;
    pToken = strtok_r(nullptr, ",", &pSave);
    if (nullptr == pToken) {
      LOG(WARNING) << boost::format("can not extract destination from (%s)") % pLog;
      continue;
    }

    auto dst = strtoul(pToken, nullptr, 10);
    CHECK(dst <= std::numeric_limits<VID_T>::max()) << "dst: " << src << " exceed max value";
    buffer[count].dst_ = dst;

    if (3 == valid_splits) {
      pLog = pToken;
      pToken = strtok_r(nullptr, ",", &pSave);

      if (false == decoder(&(buffer[count].edata_), pToken)) {
        LOG(WARNING) << boost::format("can not decode EdgeData from (%s)") % pLog;
        continue;
      }
    }

    ++total_count;
    ++count;

    if (count >= HUGESIZE) {
      if (false == callback(buffer.get(), count)) {
        LOG(ERROR) << boost::format("blockcallback failed");
      }
      count = 0;
    }
  }
  if (0 != count) {
    if (false == callback(buffer.get(), count)) {
      LOG(ERROR) << boost::format("blockcallback failed");
    }
    count = 0;
  }

  return total_count;
}

/*
 * \brief vertex_parser_t,  parse from input stream, call user provide function when buffer if full
 *                        vertex_parser_t must be reentrant
 *
 * \param fin,          input stream
 * \param callback      user provide callback function for take edges away
 *
 * \return:
 *      >=0 vertices parse from fin
 *      <0  something wrong happened
 *
 *
 *
 * \brief vertex_blockcallback_t, user provide function to deal with extracted edges
 *                  blockcallback_t must be reentrant
 *
 * \param buffer,   vertex_unit_t's buffer, input
 * \param lenth,    buffer's size
 *
 * \return true --  continue parse stage, false -- abort parse
 *
 **/
template <typename VertexData>
using vertex_blockcallback_t = std::function<bool(vertex_unit_t<VertexData>*, size_t)>;

template <typename STREAM_T, typename VertexData>
using vertex_parser_t = std::function<ssize_t(STREAM_T&, vertex_blockcallback_t<VertexData>, decoder_t<VertexData>)>;

// build-in parsers

/**
 * @brief parser for csv format
 * @tparam STREAM_T
 * @tparam VertexData
 * @param fin
 * @param callback
 * @param decoder
 * @return
 */
template <typename STREAM_T, typename VertexData>
ssize_t vertex_csv_parser(STREAM_T& fin, vertex_blockcallback_t<VertexData> callback, decoder_t<VertexData> decoder) {
  ssize_t total_count = 0;
  size_t  count = 0;

  char* pSave  = nullptr;
  char* pToken = nullptr;
  char* pLog   = nullptr;
  std::unique_ptr<char[]> sInput(new char[HUGESIZE]);
  std::unique_ptr< vertex_unit_t<VertexData>[] > buffer(new vertex_unit_t<VertexData>[HUGESIZE]);

  while (fin.good() && (false == fin.eof())) {
    fin.getline(sInput.get(), HUGESIZE);
    if ((false == fin.good()) || ('\0' == sInput[0])) {
      continue;
    }

    pLog   = sInput.get();
    pToken = strtok_r(sInput.get(), ",", &pSave);
    if (nullptr == pToken) {
      LOG(WARNING) << boost::format("can not extract source from (%s)") % pLog;
      continue;
    }
    buffer[count].vid_ = strtoul(pToken, nullptr, 10);

    if (false == decoder(&(buffer[count].vdata_), pSave)) {
      LOG(WARNING) << boost::format("can not decode VertexData from (%s)") % pLog;
      continue;
    }

    ++total_count;
    ++count;

    if (count >= HUGESIZE) {
      if (false == callback(buffer.get(), count)) {
        LOG(ERROR) << boost::format("vertex_blockcallback failed");
      }
      count = 0;
    }
  }
  if (0 != count) {
    if (false == callback(buffer.get(), count)) {
      LOG(ERROR) << boost::format("vertex_blockcallback failed");
    }
    count = 0;
  }

  return total_count;
}

}  // namespace plato

#endif

