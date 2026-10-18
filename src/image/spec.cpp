/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/image/types.hpp"

#include <stdexcept>

namespace flagcv {

std::size_t depth_size_in_bytes(flagcvPixelDepth_t depth) {
  switch (depth) {
    case FLAGCV_DEPTH_8U:
    case FLAGCV_DEPTH_8S:
      return 1;
    case FLAGCV_DEPTH_16U:
    case FLAGCV_DEPTH_16S:
    case FLAGCV_DEPTH_16F:
      return 2;
    case FLAGCV_DEPTH_32S:
    case FLAGCV_DEPTH_32F:
      return 4;
    case FLAGCV_DEPTH_64F:
      return 8;
  }
  throw std::invalid_argument("unknown pixel depth");
}

bool is_integer_depth(flagcvPixelDepth_t depth) {
  switch (depth) {
    case FLAGCV_DEPTH_8U:
    case FLAGCV_DEPTH_8S:
    case FLAGCV_DEPTH_16U:
    case FLAGCV_DEPTH_16S:
    case FLAGCV_DEPTH_32S:
      return true;
    case FLAGCV_DEPTH_16F:
    case FLAGCV_DEPTH_32F:
    case FLAGCV_DEPTH_64F:
      return false;
  }
  throw std::invalid_argument("unknown pixel depth");
}

std::size_t image_size_in_bytes(const ImageSpec& spec) {
  std::size_t total = 0;
  for (std::int32_t index = 0; index < spec.plane_count; ++index) {
    const PlaneSpec& plane = spec.planes[static_cast<std::size_t>(index)];
    const std::size_t extent =
        static_cast<std::size_t>(plane.offset_in_bytes) +
        static_cast<std::size_t>(plane.row_pitch_in_bytes) *
            static_cast<std::size_t>(plane.height);
    if (extent > total) {
      total = extent;
    }
  }
  return total;
}

bool same_layout(const ImageSpec& left, const ImageSpec& right) {
  return left.width == right.width && left.height == right.height &&
         left.depth == right.depth && left.channels == right.channels;
}

}  // namespace flagcv
