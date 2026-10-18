/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/imgproc.hpp>

#include "src/image/types.hpp"
#include "src/runtime/immediate_cache.hpp"

#include <cstddef>

namespace flagcv {
namespace {

ImageSpec to_spec(const GpuMat& image, std::int64_t uid) {
  ImageSpec spec;
  spec.uid = uid;
  spec.width = image.width();
  spec.height = image.height();
  spec.depth = image.depth();
  spec.channels = image.channels();
  spec.format = image.format();
  spec.plane_count = 1;
  spec.planes[0].row_pitch_in_bytes = image.row_pitch();
  spec.planes[0].width = image.width();
  spec.planes[0].height = image.height();
  spec.planes[0].channels = image.channels();
  return spec;
}

/* Stable UIDs; an immediate call always binds the same port layout. */
constexpr std::int64_t kSrc1Uid = 1;
constexpr std::int64_t kSrc2Uid = 2;
constexpr std::int64_t kMaskUid = 3;
constexpr std::int64_t kDestinationUid = 4;

}  // namespace

std::size_t GpuMat::element_size() const {
  return depth_size_in_bytes(depth_);
}

std::int64_t GpuMat::row_pitch() const {
  if (row_pitch_ != 0) {
    return row_pitch_;
  }
  return static_cast<std::int64_t>(width_) *
         static_cast<std::int64_t>(channels_) *
         static_cast<std::int64_t>(element_size());
}

bool GpuMat::same_layout_as(const GpuMat& other) const {
  return width_ == other.width_ && height_ == other.height_ &&
         depth_ == other.depth_ && channels_ == other.channels_;
}

void bitwise_and(Handle& handle,
                 const GpuMat& src1,
                 const GpuMat& src2,
                 const GpuMat& dst,
                 const GpuMat& mask,
                 flagcvStream_t stream) {
  if (src1.empty() || src2.empty() || dst.empty()) {
    throw Error(FLAGCV_STATUS_INVALID_VALUE,
                "bitwise_and requires non-empty operands");
  }

  OperationSpec operation;
  operation.operation = FLAGCV_OPERATION_ELEMENTWISE;
  operation.name = "bitwise_and";
  operation.attributes["mode"] =
      static_cast<std::int64_t>(FLAGCV_ELEMENTWISE_BITWISE_AND);
  operation.inputs.push_back({"src1", to_spec(src1, kSrc1Uid), false});
  operation.inputs.push_back({"src2", to_spec(src2, kSrc2Uid), false});

  const bool has_mask = !mask.empty();
  if (has_mask) {
    operation.inputs.push_back({"mask", to_spec(mask, kMaskUid), true});
  }
  operation.outputs.push_back(
      {"destination", to_spec(dst, kDestinationUid), false});
  operation.finalized = true;

  /*
   * Operand validation lives in lowering, so the immediate and pipeline paths
   * cannot drift apart on what they accept.
   */
  std::vector<flagcvBinding_t> bindings{
      {kSrc1Uid, src1.data()},
      {kSrc2Uid, src2.data()},
      {kDestinationUid, dst.data()},
  };
  if (has_mask) {
    bindings.push_back({kMaskUid, mask.data()});
  }

  execute_immediate(handle, operation, bindings, stream);
}

}  // namespace flagcv
