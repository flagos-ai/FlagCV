/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_RUNTIME_BACKEND_PLUGIN_HPP_
#define FLAGCV_SRC_RUNTIME_BACKEND_PLUGIN_HPP_

#include "backends/backend.hpp"

#include <string>

namespace flagcv {

// Owns a DSO and one device context. Kernels must be unloaded first.
class BackendPlugin {
 public:
  BackendPlugin(const std::string& name, int device);
  ~BackendPlugin();
  BackendPlugin(const BackendPlugin&) = delete;
  BackendPlugin& operator=(const BackendPlugin&) = delete;

  const BackendApi& api() const { return *api_; }
  void* context() const { return context_; }

 private:
  void* library_ = nullptr;
  const BackendApi* api_ = nullptr;
  void* context_ = nullptr;
};

}  // namespace flagcv

#endif  // FLAGCV_SRC_RUNTIME_BACKEND_PLUGIN_HPP_
