/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/nvidia/dispatch/kernel_selection.hpp"

#include <stdexcept>

namespace flagcv::nvidia {

std::int32_t total_components_per_row(const LoweredOperation& lowered) {
  for (const auto& entry : lowered.images) {
    if (entry.first == "destination") {
      return entry.second.width * entry.second.channels;
    }
  }
  throw std::invalid_argument(
      "lowered operation has no destination image");
}

}  // namespace flagcv::nvidia
