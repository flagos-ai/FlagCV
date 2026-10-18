/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_HANDLE_HPP_
#define FLAGCV_HANDLE_HPP_

#include <memory>
#include <stdexcept>
#include <string>

namespace flagcv {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message) : std::runtime_error(message) {}
};

/*
 * Binds one backend and device, and owns the compiled-kernel cache shared by
 * every operator call made through it.
 *
 * The cache is why immediate mode is viable: the first call with a given
 * operand layout compiles, the rest only launch. Backends are loaded by name
 * at runtime, so adding a platform does not relink the caller.
 *
 * Not thread-safe; use one per thread.
 */
class Handle {
 public:
  explicit Handle(const std::string& backend = "nvidia", int device = 0);
  ~Handle();

  Handle(Handle&&) noexcept;
  Handle& operator=(Handle&&) noexcept;
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  const std::string& backend() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace flagcv

#endif  // FLAGCV_HANDLE_HPP_
