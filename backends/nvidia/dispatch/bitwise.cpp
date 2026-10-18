/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/nvidia/dispatch/kernel_selection.hpp"

#include <cstdint>

namespace flagcv::nvidia {
namespace {

bool is_byte_depth(flagcvPixelDepth_t depth) {
  return depth == FLAGCV_DEPTH_8U || depth == FLAGCV_DEPTH_8S;
}

/*
 * The packed kernel reinterprets four 8-bit components as one int32, so every
 * participating row must start 4-byte aligned and hold a whole number of
 * words. A row pitch that is not a multiple of four would make the second row
 * start mid-word.
 */
bool row_is_packable(const ImageSpec& image) {
  const std::int64_t pitch = image.planes[0].row_pitch_in_bytes;
  return pitch % 4 == 0;
}

bool all_rows_packable(const LoweredOperation& lowered) {
  for (const auto& entry : lowered.images) {
    /* The mask stays byte-addressed, so its pitch is unconstrained. */
    if (entry.first == "mask") {
      continue;
    }
    if (!is_byte_depth(entry.second.depth) || !row_is_packable(entry.second)) {
      return false;
    }
  }
  return true;
}

}  // namespace

/*
 * Chooses between the platform packed kernel and the common kernel.
 *
 * Platform registration is exclusive: once this function selects the packed
 * kernel, a later failure does not silently fall back. The decision therefore
 * happens here at build time, where the operand layout is fully known, rather
 * than inside the kernel.
 */
KernelSelection select_bitwise_kernel(const LoweredOperation& lowered) {
  KernelSelection selection;

  const std::int32_t components_per_row = total_components_per_row(lowered);
  const bool packable =
      all_rows_packable(lowered) && components_per_row % 4 == 0;

  if (!packable) {
    selection.registry = KernelRegistry::kCommon;
    selection.kernel_operation = lowered.kernel_operation;
    selection.reason = "operand layout is not int32-packable";
    return selection;
  }

  selection.registry = KernelRegistry::kPlatform;
  selection.kernel_operation = lowered.kernel_operation;
  selection.packed_words_per_row = components_per_row / 4;
  selection.reason = "8-bit operands with 4-byte aligned rows";
  return selection;
}

}  // namespace flagcv::nvidia
