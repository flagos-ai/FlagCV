/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_REFERENCE_CPU_FLIP_HPP_
#define FLAGCV_REFERENCE_CPU_FLIP_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace flagcv {
namespace reference {

/*
 * Scalar flip over raw bytes, following cv::flip: flip_code == 0 reverses
 * rows, > 0 reverses columns, < 0 both. Pitches are in bytes. Pixels are
 * copied whole, so the result is bit-exact for every depth. Bytes of `dst`
 * past each row (padding) are left untouched.
 */
inline void flip(const std::uint8_t* src, std::int64_t src_pitch,
                 std::uint8_t* dst, std::int64_t dst_pitch, int width,
                 int height, std::size_t pixel_bytes, int flip_code) {
  const bool flip_x = flip_code != 0;
  const bool flip_y = flip_code <= 0;
  for (int y = 0; y < height; ++y) {
    const int sy = flip_y ? height - 1 - y : y;
    for (int x = 0; x < width; ++x) {
      const int sx = flip_x ? width - 1 - x : x;
      std::memcpy(dst + y * dst_pitch + x * pixel_bytes,
                  src + sy * src_pitch + sx * pixel_bytes, pixel_bytes);
    }
  }
}

}  // namespace reference
}  // namespace flagcv

#endif  // FLAGCV_REFERENCE_CPU_FLIP_HPP_
