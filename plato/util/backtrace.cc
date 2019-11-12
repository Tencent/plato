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
 * Copyright 2017 ScyllaDB
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

#include "backtrace.h"

#include <link.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

#include <errno.h>
#include <string.h>

namespace plato {

static int dl_iterate_phdr_callback(struct dl_phdr_info *info, size_t /* size */, void *data)
{
  std::size_t total_size{0};
  for (int i = 0; i < info->dlpi_phnum; i++) {
    const auto hdr = info->dlpi_phdr[i];

    // Only account loadable, executable (text) segments
    if (hdr.p_type == PT_LOAD && (hdr.p_flags & PF_X) == PF_X) {
      total_size += hdr.p_memsz;
    }
  }

  reinterpret_cast<std::vector<shared_object>*>(data)->push_back({info->dlpi_name, info->dlpi_addr, info->dlpi_addr + total_size});

  return 0;
}

static std::vector<shared_object> enumerate_shared_objects() {
  std::vector<shared_object> shared_objects;
  dl_iterate_phdr(dl_iterate_phdr_callback, &shared_objects);

  return shared_objects;
}

static const std::vector<shared_object> shared_objects{enumerate_shared_objects()};
static const shared_object uknown_shared_object{"", 0, std::numeric_limits<uintptr_t>::max()};

bool operator==(const frame& a, const frame& b) {
  return a.so == b.so && a.addr == b.addr;
}

frame decorate(uintptr_t addr) {
  char** s = backtrace_symbols((void**)&addr, 1);
  std::string symbol(*s);
  free(s);

  // If the shared-objects are not enumerated yet, or the enumeration
  // failed return the addr as-is with a dummy shared-object.
  if (shared_objects.empty()) {
    return {&uknown_shared_object, addr, std::move(symbol)};
  }

  auto it = std::find_if(shared_objects.begin(), shared_objects.end(), [&] (const shared_object& so) {
    return addr >= so.begin && addr < so.end;
  });

  // Unidentified addresses are assumed to originate from the executable.
  auto& so = it == shared_objects.end() ? shared_objects.front() : *it;
  return {&so, addr - so.begin, std::move(symbol)};
}

saved_backtrace current_backtrace() noexcept {
  saved_backtrace::vector_type v;
  backtrace([&] (frame f) {
    if (v.size() < v.capacity()) {
      v.emplace_back(std::move(f));
    }
  });
  return saved_backtrace(std::move(v));
}

size_t saved_backtrace::hash() const {
  size_t h = 0;
  for (auto f : _frames) {
    h = ((h << 5) - h) ^ (f.so->begin + f.addr);
  }
  return h;
}

std::ostream& operator<<(std::ostream& out, const saved_backtrace& b) {
  for (auto f : b._frames) {
    out << "  ";
    if (!f.so->name.empty()) {
      out << f.so->name << "+";
    }
    out << boost::format("0x%x\n") % f.addr;
  }
  return out;
}


// Installs handler for Signal which ensures that Func is invoked only once
// in the whole program and that after it is invoked the default handler is restored.
template<int Signal, void(*Func)()>
void install_oneshot_signal_handler() {
  static bool handled = false;
  static std::mutex lock;

  struct sigaction sa;
  sa.sa_sigaction = [](int sig, siginfo_t */* info */, void */* p */) {
    std::lock_guard<std::mutex> g(lock);
    if (!handled) {
      handled = true;
      Func();
      signal(sig, SIG_DFL);
    }
  };
  sigfillset(&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO | SA_RESTART;
  if (Signal == SIGSEGV) {
    sa.sa_flags |= SA_ONSTACK;
  }
  auto r = ::sigaction(Signal, &sa, nullptr);
  if (r == -1) {
    throw std::system_error();
  }
}

static void print_with_backtrace(backtrace_buffer& buf) noexcept {
  buf.append(".\nBacktrace:\n");
  buf.append_backtrace();
  buf.flush();
}

static void print_with_backtrace(const char* cause) noexcept {
  backtrace_buffer buf;
  buf.append(cause);
  print_with_backtrace(buf);
}

static void sigsegv_action() noexcept {
  print_with_backtrace("Segmentation fault");
}

static void sigabrt_action() noexcept {
  print_with_backtrace("Aborting");
}

void install_oneshot_signal_handlers() {
  // Mask most, to prevent threads (esp. dpdk helper threads)
  // from servicing a signal.  Individual reactors will unmask signals
  // as they become prepared to handle them.
  //
  // We leave some signals unmasked since we don't handle them ourself.

  sigset_t sigs;
  sigfillset(&sigs);
  for (auto sig : {SIGHUP, SIGQUIT, SIGILL, SIGABRT, SIGFPE, SIGSEGV, SIGALRM, SIGCONT, SIGSTOP, SIGTSTP, SIGTTIN, SIGTTOU}) {
    sigdelset(&sigs, sig);
  }
  pthread_sigmask(SIG_BLOCK, &sigs, nullptr);

  install_oneshot_signal_handler<SIGSEGV, sigsegv_action>();
  install_oneshot_signal_handler<SIGABRT, sigabrt_action>();
}


}
