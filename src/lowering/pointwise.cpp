/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <limits>
#include <stdexcept>

#include "src/lowering/helpers.hpp"
#include "src/lowering/lowering.hpp"

namespace flagcv {
namespace {
void require_safe_alias(const GpuMat& src, const GpuMat& dst) {
  // A work item reads its own element before storing it. Other overlaps can
  // race between programs, including views with unequal pitches.
  if (images_overlap(src, dst) &&
      (src.data() != dst.data() || src.row_pitch() != dst.row_pitch())) {
    throw std::invalid_argument("pointwise does not support partial overlap");
  }
}
}  // namespace

LoweredOp lower_pointwise(const GpuMat& src1, const GpuMat& src2,
                          const GpuMat& dst, PointwiseMode mode) {
  const ImageSpec a = to_spec(src1, "src1");
  const ImageSpec b = to_spec(src2, "src2");
  const ImageSpec d = to_spec(dst, "dst");
  require_same_layout(a, b);
  require_same_layout(a, d);
  const bool bitwise =
      mode == PointwiseMode::kAnd || mode == PointwiseMode::kNot;
  if (a.depth != Depth::k8U && (bitwise || a.depth != Depth::k32F)) {
    throw std::invalid_argument(bitwise
                                    ? "bitwise supports 8U only"
                                    : "arithmetic supports 8U and 32F only");
  }
  if (a.channels > 4 || std::int64_t{a.width} * a.channels >
                            std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument(
        "pointwise requires 1-4 channels and an i32 row width");
  }
  require_safe_alias(src1, dst);
  require_safe_alias(src2, dst);

  LoweredOp op;
  switch (mode) {
    case PointwiseMode::kAdd:
      op.kernel = "add";
      break;
    case PointwiseMode::kSubtract:
      op.kernel = "subtract";
      break;
    case PointwiseMode::kMultiply:
      op.kernel = "multiply";
      break;
    case PointwiseMode::kAnd:
      op.kernel = "bitwise_and";
      break;
    case PointwiseMode::kNot:
      op.kernel = "bitwise_not";
      break;
  }
  op.add_image(a);
  op.add_image(b);
  op.add_image(d);
  op.add_hinted_param("width", a.width);
  op.add_param("height", a.height);
  op.add_hinted_param("src1_row_stride", a.row_stride);
  op.add_hinted_param("src2_row_stride", b.row_stride);
  op.add_hinted_param("dst_row_stride", d.row_stride);
  op.add_constexpr("CHANNELS", a.channels);
  op.add_constexpr("MODE", static_cast<int>(mode));
  return op;
}

}  // namespace flagcv
