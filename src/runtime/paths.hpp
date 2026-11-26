/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_RUNTIME_PATHS_HPP_
#define FLAGCV_SRC_RUNTIME_PATHS_HPP_

#include <string>

namespace flagcv {

/*
 * Where the runtime finds its non-compiled parts.
 *
 * Data directory: holds kernels/, compiler/ and backends/<platform>/ with each
 * platform's compiler.py and kernel registry. Plugin directory: holds
 * libflagcv_<platform>.so.
 *
 * Each is resolved as: an environment override (FLAGCV_DATA_DIR,
 * FLAGCV_PLUGIN_DIR), then the install layout relative to the directory
 * holding libflagcv (../share/flagcv and ./flagcv), then the build tree this
 * library was compiled in.
 */
const std::string& data_dir();
std::string plugin_path(const std::string& backend);

}  // namespace flagcv

#endif  // FLAGCV_SRC_RUNTIME_PATHS_HPP_
