/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/pipeline/lowering/lowering.hpp"

#include "src/pipeline/lowering/helpers.hpp"

#include <stdexcept>

namespace flagcv {
namespace {

/* Mode ordinals are consumed as a constexpr by kernels/common/bitwise.py. */
std::int64_t bitwise_mode_ordinal(flagcvElementwiseMode_t mode) {
  switch (mode) {
    case FLAGCV_ELEMENTWISE_BITWISE_AND:
      return 0;
    case FLAGCV_ELEMENTWISE_BITWISE_OR:
      return 1;
    case FLAGCV_ELEMENTWISE_BITWISE_XOR:
      return 2;
    default:
      throw std::invalid_argument("operation is not a bitwise binary mode");
  }
}

}  // namespace

LoweredOperation lower_bitwise(const OperationSpec& operation) {
  const ImageSpec& destination = require_output(operation, "destination");
  const auto mode =
      static_cast<flagcvElementwiseMode_t>(require_int64(operation, "mode"));

  /*
   * A bit-for-bit reinterpretation of float operands would silently produce
   * values the caller never asked for, so reject the depth instead.
   */
  if (!is_integer_depth(destination.depth)) {
    throw std::invalid_argument(
        "bitwise operations require an integer pixel depth");
  }

  LoweredOperation lowered;

  if (mode == FLAGCV_ELEMENTWISE_BITWISE_NOT) {
    const ImageSpec& source = require_input(operation, "source");
    require_same_extent(source, destination);
    require_same_depth(source, destination);
    lowered.kernel_operation = "bitwise_not";
    lowered.add_image("src", source);
  } else {
    lowered.kernel_operation = "bitwise_binary";
    lowered.add_parameter("mode", bitwise_mode_ordinal(mode));

    if (has_input(operation, "src2")) {
      const ImageSpec& left = require_input(operation, "src1");
      const ImageSpec& right = require_input(operation, "src2");
      require_same_extent(left, right);
      require_same_extent(left, destination);
      require_same_depth(left, right);
      require_same_depth(left, destination);
      lowered.add_image("src1", left);
      lowered.add_image("src2", right);
    } else {
      /*
       * The scalar operand is a device buffer of `channels` elements rather
       * than a compiled-in constant, so changing the scalar reuses the cached
       * artifact instead of forcing a recompile.
       */
      const ImageSpec& source = require_input(operation, "source");
      require_same_extent(source, destination);
      require_same_depth(source, destination);
      lowered.kernel_operation = "bitwise_binary_scalar";
      lowered.add_image("src", source);
      lowered.add_scalar_operand("scalar", destination.channels);
    }
  }

  /*
   * A mask narrows the store predicate instead of selecting between values,
   * which is what keeps masked-out destination pixels byte-identical to their
   * prior contents.
   */
  const bool has_mask = has_input(operation, "mask");
  lowered.add_parameter("has_mask", has_mask ? 1 : 0);
  if (has_mask) {
    const ImageSpec& mask = require_input(operation, "mask");
    require_same_extent(mask, destination);
    if (mask.channels != 1 || mask.depth != FLAGCV_DEPTH_8U) {
      throw std::invalid_argument(
          "bitwise mask must be an 8-bit single-channel image");
    }
    lowered.add_image("mask", mask);
  }

  lowered.add_image("destination", destination);
  lowered.add_parameter("channels", destination.channels);
  lowered.add_parameter("width", destination.width);
  lowered.add_parameter("height", destination.height);
  return lowered;
}

}  // namespace flagcv
