/*
 * Copyright (c) 2025-2026 BAAI. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLAGCV_IMGPROC_HPP_
#define FLAGCV_IMGPROC_HPP_

#include <flagcv/flagcv.hpp>

#include <cstdint>

/*
 * Immediate-mode operations for single calls. Execution goes through the same
 * lowering and kernels as the pipeline API; this is a convenience surface, not
 * a second implementation.
 *
 * Each call resolves a cached single-operation executable keyed by operand
 * metadata, so only the first call with a given layout pays build cost. Work
 * is enqueued on `stream` and is not synchronized; the caller decides when to
 * wait.
 */
namespace flagcv {

/*
 * Device image view. FlagCV never allocates or frees device memory on behalf
 * of the caller, so a view only pairs caller-owned storage with its layout and
 * deliberately does not own its buffer.
 *
 * A zero row_pitch means densely packed rows.
 */
class GpuMat {
 public:
  GpuMat() = default;

  GpuMat(void* data,
         std::int32_t width,
         std::int32_t height,
         flagcvPixelDepth_t depth,
         std::int32_t channels,
         std::int64_t row_pitch_in_bytes = 0,
         flagcvColorFormat_t format = FLAGCV_FORMAT_GENERIC)
      : data_(data),
        width_(width),
        height_(height),
        depth_(depth),
        channels_(channels),
        row_pitch_(row_pitch_in_bytes),
        format_(format) {}

  [[nodiscard]] void* data() const { return data_; }
  [[nodiscard]] std::int32_t width() const { return width_; }
  [[nodiscard]] std::int32_t height() const { return height_; }
  [[nodiscard]] flagcvPixelDepth_t depth() const { return depth_; }
  [[nodiscard]] std::int32_t channels() const { return channels_; }
  [[nodiscard]] flagcvColorFormat_t format() const { return format_; }
  [[nodiscard]] bool empty() const { return data_ == nullptr; }

  /* Bytes per channel value. */
  [[nodiscard]] std::size_t element_size() const;
  /* Bytes per row, resolving a zero pitch to the densely packed value. */
  [[nodiscard]] std::int64_t row_pitch() const;

  /* True when extent, depth and channel count all agree. */
  [[nodiscard]] bool same_layout_as(const GpuMat& other) const;

 private:
  void* data_ = nullptr;
  std::int32_t width_ = 0;
  std::int32_t height_ = 0;
  flagcvPixelDepth_t depth_ = FLAGCV_DEPTH_8U;
  std::int32_t channels_ = 0;
  std::int64_t row_pitch_ = 0;
  flagcvColorFormat_t format_ = FLAGCV_FORMAT_GENERIC;
};

/*
 * Computes the per-element bitwise conjunction of two images.
 *
 *   src1, src2  inputs of identical extent, depth and channel count
 *   dst         output, which must already have a matching layout because
 *               FlagCV does not allocate device memory
 *   mask        optional 8-bit single-channel operation mask
 *   stream      target stream; null selects the backend default
 *
 * Only integer depths are accepted; a floating-point input is rejected rather
 * than reinterpreted bit-for-bit.
 *
 * Where mask is non-empty and its value is zero, the corresponding dst pixel
 * is left unmodified, so a masked call must pass an initialized dst.
 *
 * Throws flagcv::Error when the operands are inconsistent.
 */
void bitwise_and(Handle& handle,
                 const GpuMat& src1,
                 const GpuMat& src2,
                 const GpuMat& dst,
                 const GpuMat& mask = GpuMat(),
                 flagcvStream_t stream = nullptr);

}  // namespace flagcv

#endif  // FLAGCV_IMGPROC_HPP_
