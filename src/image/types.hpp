/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_IMAGE_TYPES_HPP_
#define FLAGCV_SRC_IMAGE_TYPES_HPP_

#include <flagcv/flagcv.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace flagcv {

struct PlaneSpec {
  std::int64_t offset_in_bytes = 0;
  std::int64_t row_pitch_in_bytes = 0;
  std::int32_t width = 0;
  std::int32_t height = 0;
  std::int32_t channels = 0;
};

struct RegionSpec {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t width = 0;
  std::int32_t height = 0;
};

/*
 * Platform-neutral image data model. Unlike a dense strided tensor, an image
 * carries colour semantics and per-plane pitch, because imaging hardware
 * commonly requires row padding and chroma subsampling.
 */
struct ImageSpec {
  std::int64_t uid = 0;
  std::int32_t width = 0;
  std::int32_t height = 0;
  flagcvPixelDepth_t depth = FLAGCV_DEPTH_8U;
  std::int32_t channels = 0;
  flagcvColorFormat_t format = FLAGCV_FORMAT_GENERIC;
  std::int32_t plane_count = 1;
  std::array<PlaneSpec, FLAGCV_MAX_PLANES> planes{};
  std::optional<RegionSpec> region;
  bool is_virtual = false;
  std::int64_t alignment = 16;
  std::string name;
};

std::size_t depth_size_in_bytes(flagcvPixelDepth_t depth);
bool is_integer_depth(flagcvPixelDepth_t depth);

/* Total bytes spanned including row padding and all planes. */
std::size_t image_size_in_bytes(const ImageSpec& spec);

/* True when extent, depth and channel count all agree. */
bool same_layout(const ImageSpec& left, const ImageSpec& right);

}  // namespace flagcv

#endif  // FLAGCV_SRC_IMAGE_TYPES_HPP_
