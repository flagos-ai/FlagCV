/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <limits>
#include <stdexcept>

#include "src/lowering/helpers.hpp"
#include "src/lowering/lowering.hpp"

namespace flagcv {

LoweredOp lower_flip(const GpuMat& src, const GpuMat& dst, int flip_code) {
  const ImageSpec s = to_spec(src, "src");
  const ImageSpec d = to_spec(dst, "dst");
  require_same_layout(s, d);

  if (s.channels > 4) {
    throw std::invalid_argument("flip supports 1 to 4 channels");
  }
  // The kernel indexes a row's channel values with 32-bit arithmetic.
  if (static_cast<std::int64_t>(s.width) * s.channels >
      std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument("flip row is too wide");
  }
  // Each output pixel reads a different input pixel, so in place would race.
  if (images_overlap(src, dst)) {
    throw std::invalid_argument(
        "flip does not support overlapping src and dst");
  }

  LoweredOp op;
  op.kernel = "flip";
  op.add_image(s);
  op.add_image(d);
  // width bounds the vectorised row axis and the strides place row starts;
  // height only masks whole rows, so it gets no divisibility specialisation.
  op.add_hinted_param("width", s.width);
  op.add_param("height", s.height);
  op.add_hinted_param("src_row_stride", s.row_stride);
  op.add_hinted_param("dst_row_stride", d.row_stride);
  op.add_constexpr("CHANNELS", s.channels);
  // OpenCV: 0 flips around the x-axis, > 0 around the y-axis, < 0 both.
  op.add_constexpr("FLIP_X", flip_code != 0 ? 1 : 0);
  op.add_constexpr("FLIP_Y", flip_code <= 0 ? 1 : 0);
  return op;
}

}  // namespace flagcv
