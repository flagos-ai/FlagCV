/*
 * Copyright (c) 2025-2026 BAAI. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLAGCV_FLAGCV_HPP_
#define FLAGCV_FLAGCV_HPP_

#include <flagcv/flagcv.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagcv {

/* Carries the status code alongside the runtime's thread-local diagnostic. */
class Error : public std::runtime_error {
 public:
  Error(flagcvStatus_t status, const std::string& message)
      : std::runtime_error(message), status_(status) {}

  [[nodiscard]] flagcvStatus_t status() const { return status_; }

 private:
  flagcvStatus_t status_;
};

inline void check(flagcvStatus_t status) {
  if (status != FLAGCV_STATUS_SUCCESS) {
    const char* detail = flagcvGetLastErrorString();
    const char* summary = flagcvGetErrorString(status);
    std::string message(summary != nullptr ? summary : "flagcv error");
    if (detail != nullptr && *detail != '\0') {
      message += ": ";
      message += detail;
    }
    throw Error(status, message);
  }
}

/* Owns a runtime context bound to one backend and device. */
class Handle {
 public:
  Handle() { check(flagcvCreate(&value_)); }

  explicit Handle(std::string_view backend_name, std::int32_t device = 0) {
    const std::string owned_name(backend_name);
    check(flagcvCreateWithBackendName(owned_name.c_str(), device, &value_));
  }

  ~Handle() { reset(); }

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  Handle(Handle&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  Handle& operator=(Handle&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

  [[nodiscard]] std::string_view backend_name() const {
    const char* result = nullptr;
    check(flagcvGetBackendName(value_, &result));
    return result != nullptr ? std::string_view(result) : std::string_view();
  }

  [[nodiscard]] std::string_view target_fingerprint() const {
    const char* result = nullptr;
    check(flagcvGetTargetFingerprint(value_, &result));
    return result != nullptr ? std::string_view(result) : std::string_view();
  }

  void set_compiler(const std::string& executable,
                    const std::string& compiler,
                    const std::string& cache_directory) {
    check(flagcvSetCompilerConfig(value_,
                                  executable.c_str(),
                                  compiler.c_str(),
                                  cache_directory.c_str()));
  }

  [[nodiscard]] flagcvHandle_t get() const { return value_; }

 private:
  void reset() {
    if (value_ != nullptr) {
      flagcvDestroy(value_);
      value_ = nullptr;
    }
  }

  flagcvHandle_t value_ = nullptr;
};

}  // namespace flagcv

#endif  // FLAGCV_FLAGCV_HPP_
