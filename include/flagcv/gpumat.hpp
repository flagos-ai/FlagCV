/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_GPUMAT_HPP_
#define FLAGCV_GPUMAT_HPP_

#include <flagcv/types.hpp>

#include <cstddef>
#include <cstdint>

namespace flagcv {

/*
 * Device image view: the operand type for every operator.
 *
 * Named after cv::cuda::GpuMat so OpenCV code reads the same way, but unlike
 * GpuMat it does not own its storage. FlagCV never allocates or frees device
 * memory, so this only pairs caller-owned memory with its layout, and `dst`
 * must already be allocated with a matching layout.
 *
 * `row_pitch` is in BYTES and is independent of width * channels * depth_size,
 * because imaging hardware commonly pads rows for DMA alignment. Zero means
 * densely packed. Source and destination pitches need not agree.
 */
class GpuMat {
 public:
  GpuMat() = default;

  GpuMat(void* data,
         std::int32_t width,
         std::int32_t height,
         Depth depth,
         std::int32_t channels,
         std::int64_t row_pitch = 0)
      : data_(data),
        width_(width),
        height_(height),
        depth_(depth),
        channels_(channels),
        row_pitch_(row_pitch) {}

  void* data() const { return data_; }
  std::int32_t width() const { return width_; }
  std::int32_t height() const { return height_; }
  Depth depth() const { return depth_; }
  std::int32_t channels() const { return channels_; }
  bool empty() const { return data_ == nullptr; }

  /* Bytes per row, resolving a zero pitch to the densely packed value. */
  std::int64_t row_pitch() const;

  /*
   * Row stride counted in ELEMENTS.
   *
   * Kernels address rows with this, not the byte pitch: a kernel pointer is
   * typed by pixel depth, so its arithmetic advances in elements. The public
   * API stays in bytes and this is the only conversion point.
   */
  std::int64_t row_stride() const;

 private:
  void* data_ = nullptr;
  std::int32_t width_ = 0;
  std::int32_t height_ = 0;
  Depth depth_ = Depth::k8U;
  std::int32_t channels_ = 0;
  std::int64_t row_pitch_ = 0;
};

}  // namespace flagcv

#endif  // FLAGCV_GPUMAT_HPP_
