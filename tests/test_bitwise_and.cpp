/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

/*
 * Lowering-level checks for bitwise_and. These exercise operand validation and
 * the lowered form without touching a device, so they run anywhere.
 */

#include "src/pipeline/lowering/lowering.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
  if (!condition) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
}

template <typename Callable>
void expect_throws(Callable&& callable, const char* what) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return;
  }
  std::printf("FAIL %s (expected a rejection)\n", what);
  ++failures;
}

flagcv::ImageSpec make_image(std::int64_t uid,
                             std::int32_t width,
                             std::int32_t height,
                             flagcvPixelDepth_t depth,
                             std::int32_t channels) {
  flagcv::ImageSpec spec;
  spec.uid = uid;
  spec.width = width;
  spec.height = height;
  spec.depth = depth;
  spec.channels = channels;
  spec.plane_count = 1;
  spec.planes[0].width = width;
  spec.planes[0].height = height;
  spec.planes[0].channels = channels;
  spec.planes[0].row_pitch_in_bytes =
      static_cast<std::int64_t>(width) * channels *
      static_cast<std::int64_t>(flagcv::depth_size_in_bytes(depth));
  return spec;
}

flagcv::OperationSpec make_and_operation(flagcvPixelDepth_t depth,
                                         std::int32_t channels,
                                         bool with_mask) {
  constexpr std::int32_t kWidth = 64;
  constexpr std::int32_t kHeight = 32;

  flagcv::OperationSpec operation;
  operation.operation = FLAGCV_OPERATION_ELEMENTWISE;
  operation.name = "bitwise_and";
  operation.attributes["mode"] =
      static_cast<std::int64_t>(FLAGCV_ELEMENTWISE_BITWISE_AND);
  operation.inputs.push_back(
      {"src1", make_image(1, kWidth, kHeight, depth, channels), false});
  operation.inputs.push_back(
      {"src2", make_image(2, kWidth, kHeight, depth, channels), false});
  if (with_mask) {
    operation.inputs.push_back(
        {"mask", make_image(3, kWidth, kHeight, FLAGCV_DEPTH_8U, 1), true});
  }
  operation.outputs.push_back(
      {"destination", make_image(4, kWidth, kHeight, depth, channels), false});
  operation.finalized = true;
  return operation;
}

const std::int64_t* find_parameter(const flagcv::LoweredOperation& lowered,
                                   const std::string& name) {
  for (const auto& entry : lowered.parameters) {
    if (entry.first == name) {
      return &entry.second;
    }
  }
  return nullptr;
}

bool has_image(const flagcv::LoweredOperation& lowered,
               const std::string& name) {
  for (const auto& entry : lowered.images) {
    if (entry.first == name) {
      return true;
    }
  }
  return false;
}

void unmasked_lowers_to_binary_kernel() {
  const auto lowered =
      flagcv::lower_operation(make_and_operation(FLAGCV_DEPTH_8U, 3, false));

  expect(lowered.kernel_operation == "bitwise_binary",
         "unmasked call selects the binary kernel");
  expect(has_image(lowered, "src1") && has_image(lowered, "src2") &&
             has_image(lowered, "destination"),
         "binary kernel binds both sources and the destination");
  expect(!has_image(lowered, "mask"), "unmasked call binds no mask");

  const std::int64_t* has_mask = find_parameter(lowered, "has_mask");
  expect(has_mask != nullptr && *has_mask == 0, "has_mask is zero");

  const std::int64_t* mode = find_parameter(lowered, "mode");
  expect(mode != nullptr && *mode == 0, "AND lowers to mode ordinal 0");

  const std::int64_t* channels = find_parameter(lowered, "channels");
  expect(channels != nullptr && *channels == 3, "channel count is forwarded");
}

void masked_call_binds_the_mask() {
  const auto lowered =
      flagcv::lower_operation(make_and_operation(FLAGCV_DEPTH_8U, 3, true));

  expect(has_image(lowered, "mask"), "masked call binds the mask");
  const std::int64_t* has_mask = find_parameter(lowered, "has_mask");
  expect(has_mask != nullptr && *has_mask == 1, "has_mask is one");
}

void float_depth_is_rejected() {
  expect_throws(
      [] {
        flagcv::lower_operation(
            make_and_operation(FLAGCV_DEPTH_32F, 1, false));
      },
      "float depth is rejected rather than reinterpreted");
}

void mismatched_extent_is_rejected() {
  auto operation = make_and_operation(FLAGCV_DEPTH_8U, 3, false);
  operation.inputs[1].image.width = 32;
  expect_throws([&] { flagcv::lower_operation(operation); },
                "mismatched source extent is rejected");
}

void mismatched_depth_is_rejected() {
  auto operation = make_and_operation(FLAGCV_DEPTH_8U, 3, false);
  operation.inputs[1].image.depth = FLAGCV_DEPTH_16U;
  expect_throws([&] { flagcv::lower_operation(operation); },
                "mismatched source depth is rejected");
}

void multi_channel_mask_is_rejected() {
  auto operation = make_and_operation(FLAGCV_DEPTH_8U, 3, true);
  operation.inputs[2].image.channels = 3;
  expect_throws([&] { flagcv::lower_operation(operation); },
                "a multi-channel mask is rejected");
}

void unfinalized_operation_is_rejected() {
  auto operation = make_and_operation(FLAGCV_DEPTH_8U, 1, false);
  operation.finalized = false;
  expect_throws([&] { flagcv::lower_operation(operation); },
                "an unfinalized operation is rejected");
}

}  // namespace

int main() {
  unmasked_lowers_to_binary_kernel();
  masked_call_binds_the_mask();
  float_depth_is_rejected();
  mismatched_extent_is_rejected();
  mismatched_depth_is_rejected();
  multi_channel_mask_is_rejected();
  unfinalized_operation_is_rejected();

  if (failures == 0) {
    std::printf("bitwise_and lowering: all checks passed\n");
    return 0;
  }
  std::printf("bitwise_and lowering: %d check(s) failed\n", failures);
  return 1;
}
