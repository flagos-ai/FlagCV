/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_TYPES_HPP_
#define FLAGCV_TYPES_HPP_

#include <cstddef>
#include <cstdint>

namespace flagcv {

/* Per-channel storage type. Mirrors OpenCV's CV_8U / CV_16S / ... depths. */
enum class Depth {
  k8U,
  k8S,
  k16U,
  k16S,
  k32S,
  k16F,
  k32F,
  k64F,
};

/* Bytes per channel value. */
std::size_t depth_size(Depth depth);
bool is_integer(Depth depth);

enum class Interpolation {
  kNearest,
  kLinear,
  kCubic,
  kArea,
};

enum class Border {
  kConstant,
  kReplicate,
  kReflect,
  /* Reflect without repeating the edge pixel. */
  kReflect101,
  kWrap,
};

/*
 * An opaque native backend stream: a cudaStream_t on NVIDIA, a hipStream_t
 * on AMD, an aclrtStream on Ascend. A null value means the default stream.
 * The caller owns it and keeps it alive until the enqueued work completes.
 */
using Stream = void*;

}  // namespace flagcv

#endif  // FLAGCV_TYPES_HPP_
