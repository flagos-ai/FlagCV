/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/gpumat.hpp>

#include <stdexcept>

namespace flagcv {

std::size_t depth_size(Depth depth) {
  switch (depth) {
    case Depth::k8U:
    case Depth::k8S:
      return 1;
    case Depth::k16U:
    case Depth::k16S:
    case Depth::k16F:
      return 2;
    case Depth::k32S:
    case Depth::k32F:
      return 4;
    case Depth::k64F:
      return 8;
  }
  throw std::invalid_argument("unknown pixel depth");
}

bool is_integer(Depth depth) {
  switch (depth) {
    case Depth::k8U:
    case Depth::k8S:
    case Depth::k16U:
    case Depth::k16S:
    case Depth::k32S:
      return true;
    case Depth::k16F:
    case Depth::k32F:
    case Depth::k64F:
      return false;
  }
  throw std::invalid_argument("unknown pixel depth");
}

std::int64_t GpuMat::row_pitch() const {
  if (row_pitch_ != 0) {
    return row_pitch_;
  }
  return static_cast<std::int64_t>(width_) * channels_ *
         static_cast<std::int64_t>(depth_size(depth_));
}

std::int64_t GpuMat::row_stride() const {
  const auto size = static_cast<std::int64_t>(depth_size(depth_));
  const std::int64_t pitch = row_pitch();

  /*
   * A byte pitch that is not a whole number of elements would put row starts
   * mid-element, which a typed pointer cannot address. Reject rather than
   * truncate.
   */
  if (pitch % size != 0) {
    throw std::invalid_argument(
        "row pitch must be a whole number of elements for the pixel depth");
  }
  return pitch / size;
}

}  // namespace flagcv
