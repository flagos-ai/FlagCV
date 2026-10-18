/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_BACKENDS_NVIDIA_DISPATCH_KERNEL_SELECTION_HPP_
#define FLAGCV_BACKENDS_NVIDIA_DISPATCH_KERNEL_SELECTION_HPP_

#include "src/pipeline/lowering/lowering.hpp"

#include <cstdint>
#include <string>

namespace flagcv::nvidia {

enum class KernelRegistry {
  /* backends/nvidia/kernels/registry.json */
  kPlatform,
  /* kernels/registry.json */
  kCommon,
};

struct KernelSelection {
  KernelRegistry registry = KernelRegistry::kCommon;
  std::string kernel_operation;
  /* Nonzero only when a packed platform kernel was selected. */
  std::int32_t packed_words_per_row = 0;
  /* Retained for build diagnostics. */
  std::string reason;
};

/* Components in one destination row, counting all channels. */
std::int32_t total_components_per_row(const LoweredOperation& lowered);

KernelSelection select_bitwise_kernel(const LoweredOperation& lowered);

}  // namespace flagcv::nvidia

#endif  // FLAGCV_BACKENDS_NVIDIA_DISPATCH_KERNEL_SELECTION_HPP_
