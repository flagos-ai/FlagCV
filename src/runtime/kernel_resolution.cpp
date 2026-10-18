/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/runtime/kernel_resolution.hpp"

#include <stdexcept>

namespace flagcv {

/*
 * TODO: implement two-tier resolution.
 *
 * Steps:
 *   1. Read backends/<backend_name>/kernels/registry.json. If the operation
 *      appears in its "kernels" list, return that entry.
 *   2. Otherwise read kernels/registry.json and return the common entry.
 *   3. If neither registers the operation, throw.
 *
 * Do not treat a platform entry whose constraints fail as a miss: that would
 * make platform registration non-exclusive and hide a dispatch-layer bug.
 */
ResolvedKernel resolve_kernel(std::string_view backend_name,
                              std::string_view kernel_operation) {
  (void)backend_name;
  (void)kernel_operation;
  throw std::runtime_error(
      "flagcv: kernel resolution is not implemented yet");
}

}  // namespace flagcv
