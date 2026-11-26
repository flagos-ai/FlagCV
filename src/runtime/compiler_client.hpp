/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_RUNTIME_COMPILER_CLIENT_HPP_
#define FLAGCV_SRC_RUNTIME_COMPILER_CLIENT_HPP_

#include <sys/types.h>

#include <cstdint>
#include <mutex>
#include <string>

#include "src/runtime/json.hpp"

namespace flagcv {

/*
 * Client for the external compiler (compiler/flagcv_codegen).
 *
 * One long-lived child process per host process, started on the first cache
 * miss and shared by every Handle, so the Python and Triton import is paid
 * once rather than per compile. Requests and replies are single JSON lines
 * over a socket pair bound to the child's stdin and stdout; the child's
 * stderr is inherited so compiler diagnostics reach the user.
 *
 * The interpreter is FLAGCV_PYTHON, defaulting to python3 on PATH. If the
 * child dies, the failing request reports it and the next one restarts it.
 */
class CompilerClient {
 public:
  static CompilerClient& instance();

  /*
   * Sends `body` (the members of a request object, without braces or id) and
   * returns the reply. Throws flagcv::Error if the process cannot be reached
   * or the reply carries ok=false.
   */
  json::Value request(const std::string& body);

  ~CompilerClient();

 private:
  CompilerClient() = default;
  void start();
  void stop();
  void send_line(const std::string& line);
  std::string receive_line();

  std::mutex mutex_;
  pid_t pid_ = -1;
  int fd_ = -1;
  std::string pending_;
  std::int64_t next_id_ = 1;
};

}  // namespace flagcv

#endif  // FLAGCV_SRC_RUNTIME_COMPILER_CLIENT_HPP_
