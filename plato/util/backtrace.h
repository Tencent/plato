/*
 * This file is open source software, licensed to you under the terms
 * of the Apache License, Version 2.0 (the "License").  See the NOTICE file
 * distributed with this work for additional information regarding copyright
 * ownership.  You may not use this file except in compliance with the License.
 *
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
 * Copyright 2016 ScyllaDB
 */
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

#include <execinfo.h>
#include <iosfwd>
#include <boost/container/static_vector.hpp>
#include <boost/format.hpp>
#include <mutex>

namespace plato {

struct shared_object {
  std::string name;
  uintptr_t begin;
  uintptr_t end; // C++-style, last addr + 1
};

struct frame {
  const shared_object* so;
  uintptr_t addr;
  std::string symbols;
};

bool operator==(const frame& a, const frame& b);


// If addr doesn't seem to belong to any of the provided shared objects, it
// will be considered as part of the executable.
frame decorate(uintptr_t addr);

// Invokes func for each frame passing it as argument.
template<typename Func>
void backtrace(Func&& func) noexcept(noexcept(func(frame()))) {
  constexpr size_t max_backtrace = 100;
  void* buffer[max_backtrace];
  int n = ::backtrace(buffer, max_backtrace);
  for (int i = 0; i < n; ++i) {
    auto ip = reinterpret_cast<uintptr_t>(buffer[i]);
    func(decorate(ip - 1));
  }
}

class saved_backtrace {
public:
  using vector_type = boost::container::static_vector<frame, 64>;
private:
  vector_type _frames;
public:
  saved_backtrace() = default;
  saved_backtrace(vector_type f) : _frames(std::move(f)) {}
  size_t hash() const;

  friend std::ostream& operator<<(std::ostream& out, const saved_backtrace&);

  bool operator==(const saved_backtrace& o) const {
    return _frames == o._frames;
  }

  bool operator!=(const saved_backtrace& o) const {
    return !(*this == o);
  }
};

saved_backtrace current_backtrace() noexcept;
std::ostream& operator<<(std::ostream& out, const saved_backtrace& b);

}

namespace std {

template<>
struct hash<plato::saved_backtrace> {
  size_t operator()(const plato::saved_backtrace& b) const {
    return b.hash();
  }
};

}

namespace plato {
//
// Collection of async-signal safe printing functions.
//

// Outputs string to stderr.
// Async-signal safe.
inline
void print_safe(const char *str, size_t len) noexcept {
  while (len) {
    auto result = write(STDERR_FILENO, str, len);
    if (result > 0) {
      len -= result;
      str += result;
    } else if (result == 0) {
      break;
    } else {
      if (errno == EINTR) {
        // retry
      } else {
        break; // what can we do?
      }
    }
  }
}

// Outputs string to stderr.
// Async-signal safe.
inline
void print_safe(const char *str) noexcept {
  print_safe(str, strlen(str));
}

// Fills a buffer with a zero-padded hexadecimal representation of an integer.
// For example, convert_zero_padded_hex_safe(buf, 4, uint16_t(12)) fills the buffer with "000c".
template<typename Integral>
void convert_zero_padded_hex_safe(char *buf, size_t bufsz, Integral n) noexcept {
  const char *digits = "0123456789abcdef";
  memset(buf, '0', bufsz);
  unsigned i = bufsz;
  while (n) {
    buf[--i] = digits[n & 0xf];
    n >>= 4;
  }
}

// Prints zero-padded hexadecimal representation of an integer to stderr.
// For example, print_zero_padded_hex_safe(uint16_t(12)) prints "000c".
// Async-signal safe.
template<typename Integral>
void print_zero_padded_hex_safe(Integral n) noexcept {
  static_assert(std::is_integral<Integral>::value && !std::is_signed<Integral>::value, "Requires unsigned integrals");

  char buf[sizeof(n) * 2];
  convert_zero_padded_hex_safe(buf, sizeof(buf), n);
  print_safe(buf, sizeof(buf));
}

// Fills a buffer with a decimal representation of an integer.
// The argument bufsz is the maximum size of the buffer.
// For example, print_decimal_safe(buf, 16, 12) prints "12".
template<typename Integral>
size_t convert_decimal_safe(char *buf, size_t bufsz, Integral n) noexcept {
  static_assert(std::is_integral<Integral>::value && !std::is_signed<Integral>::value, "Requires unsigned integrals");

  char tmp[sizeof(n) * 3];
  unsigned i = bufsz;
  do {
    tmp[--i] = '0' + n % 10;
    n /= 10;
  } while (n);
  memcpy(buf, tmp + i, sizeof(tmp) - i);
  return sizeof(tmp) - i;
}

// Prints decimal representation of an integer to stderr.
// For example, print_decimal_safe(12) prints "12".
// Async-signal safe.
template<typename Integral>
void print_decimal_safe(Integral n) noexcept {
  char buf[sizeof(n) * 3];
  unsigned i = sizeof(buf);
  auto len = convert_decimal_safe(buf, i, n);
  print_safe(buf, len);
}

// Accumulates an in-memory backtrace and flush to stderr eventually.
// Async-signal safe.
class backtrace_buffer {
  static constexpr unsigned _max_size = 8 << 10;
  unsigned _pos = 0;
  char _buf[_max_size];
public:
  void flush() noexcept {
    print_safe(_buf, _pos);
    _pos = 0;
  }

  void append(const char* str, size_t len) noexcept {
    if (_pos + len >= _max_size) {
      flush();
    }
    memcpy(_buf + _pos, str, len);
    _pos += len;
  }

  void append(const char* str) noexcept { append(str, strlen(str)); }

  template <typename Integral>
  void append_decimal(Integral n) noexcept {
    char buf[sizeof(n) * 3];
    auto len = convert_decimal_safe(buf, sizeof(buf), n);
    append(buf, len);
  }

  template <typename Integral>
  void append_hex(Integral ptr) noexcept {
    char buf[sizeof(ptr) * 2];
    convert_zero_padded_hex_safe(buf, sizeof(buf), ptr);
    append(buf, sizeof(buf));
  }

  void append_backtrace() noexcept {
    backtrace([this] (frame f) {
      append("  ");
      if (!f.so->name.empty()) {
        append(f.so->name.c_str(), f.so->name.size());
        append("+");
      }

      append("0x");
      append_hex(f.addr);
      append("\n");
    });
  }
};

void install_oneshot_signal_handlers();

}
