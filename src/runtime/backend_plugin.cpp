/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/runtime/backend_plugin.hpp"

#include <flagcv/handle.hpp>

#include <dlfcn.h>

#include <cstdlib>
#include <sstream>
#include <vector>

namespace flagcv {

BackendPlugin::BackendPlugin(const std::string& name, int device) {
  if (name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") !=
                          std::string::npos ||
      name.front() < 'a' || name.front() > 'z') {
    throw Error("flagcv: invalid backend name: " + name);
  }
  const auto filename = "libflagcv_backend_" + name + ".so";
  std::vector<std::string> candidates;
  if (const char* path = std::getenv("FLAGCV_BACKEND_PATH")) {
    std::istringstream paths(path);
    std::string directory;
    while (std::getline(paths, directory, ':')) {
      if (!directory.empty()) candidates.push_back(directory + "/" + filename);
    }
  }
  candidates.push_back(std::string(FLAGCV_BACKEND_INSTALL_DIR) + "/" + filename);
  candidates.push_back(filename);
  std::string diagnostic;
  for (const auto& path : candidates) {
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_) break;
    diagnostic += "\n  " + std::string(dlerror());
  }
  if (!library_) throw Error("flagcv: cannot load backend " + name + diagnostic);

  try {
    const auto get_api = reinterpret_cast<GetBackendApi>(dlsym(library_, "flagcv_backend_api"));
    if (!get_api) throw Error("flagcv: backend has no entry table: " + name);
    api_ = get_api();
    if (!api_ || api_->version != FLAGCV_BACKEND_VERSION || !api_->name ||
        name != api_->name || !api_->create_context || !api_->destroy_context ||
        !api_->device_id || !api_->load || !api_->unload || !api_->launch) {
      throw Error("flagcv: incompatible backend entry table: " + name);
    }
    BackendError error;
    if (api_->create_context(device, &context_, &error) != 0) {
      throw Error("flagcv: " + name + ": " + error.message);
    }
  } catch (...) {
    dlclose(library_);
    throw;
  }
}

BackendPlugin::~BackendPlugin() {
  if (context_) {
    BackendError error;
    (void)api_->destroy_context(context_, &error);
  }
  if (library_) dlclose(library_);
}

}  // namespace flagcv
