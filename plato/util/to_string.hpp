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

#include <set>
#include <array>
#include <deque>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <unordered_set>

namespace plato {
namespace to_string_detail {

template<typename Iterator>
static inline
std::string join(std::string delimiter, Iterator begin, Iterator end) {
  std::ostringstream oss;
  while (begin != end) {
    oss << *begin;
    ++begin;
    if (begin != end) {
      oss << delimiter;
    }
  }
  return oss.str();
}

template<typename PrintableRange>
static inline
std::string join(std::string delimiter, const PrintableRange& items) {
  return join(delimiter, items.begin(), items.end());
}

}
}

namespace std {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_set<T>& items) {
  os << "{" << plato::to_string_detail::join(", ", items) << "}";
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& items) {
  os << "{" << plato::to_string_detail::join(", ", items) << "}";
  return os;
}

template<typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& items) {
  os << "{" << plato::to_string_detail::join(", ", items) << "}";
  return os;
}

template<typename Printable>
std::ostream&operator<<(std::ostream& os, const std::vector<Printable>& items) {
  os << "{" << plato::to_string_detail::join(", ", items) << "}";
  return os;
}

template<typename Printable>
std::ostream&operator<<(std::ostream& os, const std::deque<Printable>& items) {
  os << "{" << plato::to_string_detail::join(", ", items) << "}";
  return os;
}

template<typename Printable>
std::ostream&operator<<(std::ostream& os, const std::shared_ptr<Printable>& items) {
  os << *items;
  return os;
}
}

