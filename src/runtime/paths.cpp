/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/runtime/paths.hpp"

#include <dlfcn.h>
#include <sys/stat.h>

#include <cstdlib>

#ifndef FLAGCV_SOURCE_DIR
#error "FLAGCV_SOURCE_DIR must be defined by the build"
#endif
#ifndef FLAGCV_PLUGIN_BUILD_DIR
#error "FLAGCV_PLUGIN_BUILD_DIR must be defined by the build"
#endif

namespace flagcv {

namespace {

bool exists(const std::string& path) {
  struct stat info;
  return ::stat(path.c_str(), &info) == 0;
}

/* Directory of the object holding this code: libflagcv.so, or the executable
 * when linked statically. */
std::string library_dir() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&library_dir), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  std::string path = info.dli_fname;
  auto slash = path.rfind('/');
  return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

std::string env(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string();
}

}  // namespace

const std::string& data_dir() {
  static const std::string dir = [] {
    std::string override_dir = env("FLAGCV_DATA_DIR");
    if (!override_dir.empty()) return override_dir;
    std::string installed = library_dir() + "/../share/flagcv";
    if (exists(installed + "/compiler/flagcv_codegen/main.py"))
      return installed;
    return std::string(FLAGCV_SOURCE_DIR);
  }();
  return dir;
}

std::string plugin_path(const std::string& backend) {
  const std::string file = "libflagcv_" + backend + ".so";
  std::string override_dir = env("FLAGCV_PLUGIN_DIR");
  if (!override_dir.empty()) return override_dir + "/" + file;
  std::string installed = library_dir() + "/flagcv/" + file;
  if (exists(installed)) return installed;
  return std::string(FLAGCV_PLUGIN_BUILD_DIR) + "/" + file;
}

}  // namespace flagcv
