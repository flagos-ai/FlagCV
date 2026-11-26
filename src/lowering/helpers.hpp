/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_LOWERING_HELPERS_HPP_
#define FLAGCV_SRC_LOWERING_HELPERS_HPP_

#include <flagcv/gpumat.hpp>

#include "src/image/types.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace flagcv {

/*
 * Builds the internal spec, converting the byte pitch to an element stride.
 *
 * Also rejects layouts no kernel can address safely: non-positive extents,
 * rows that overlap because the pitch is shorter than a row, and a base
 * pointer that is not aligned to the pixel depth (a typed device load from it
 * would fault).
 */
inline ImageSpec to_spec(const GpuMat& image, const std::string& name) {
  if (image.empty()) {
    throw std::invalid_argument("operand '" + name + "' must not be empty");
  }
  if (image.width() <= 0 || image.height() <= 0 || image.channels() <= 0) {
    throw std::invalid_argument("operand '" + name +
                                "' must have positive width, height and "
                                "channels");
  }
  const auto address = reinterpret_cast<std::uintptr_t>(image.data());
  if (address % depth_size(image.depth()) != 0) {
    throw std::invalid_argument("operand '" + name +
                                "' data is not aligned to its pixel depth");
  }
  ImageSpec spec;
  spec.name = name;
  spec.width = image.width();
  spec.height = image.height();
  spec.depth = image.depth();
  spec.channels = image.channels();
  spec.row_stride = image.row_stride();
  if (spec.row_stride < std::int64_t{spec.width} * spec.channels) {
    throw std::invalid_argument("operand '" + name +
                                "' row pitch is shorter than one row");
  }
  return spec;
}

/*
 * True when the byte ranges two images occupy intersect. Kernels that read
 * one pixel and write another cannot run in place, so their lowering rejects
 * overlap instead of producing an order-dependent result.
 */
inline bool images_overlap(const GpuMat& a, const GpuMat& b) {
  auto range = [](const GpuMat& image) {
    using U = std::uintptr_t;
    const U begin = reinterpret_cast<U>(image.data());
    const U row_bytes = static_cast<U>(image.width()) *
                        static_cast<U>(image.channels()) *
                        depth_size(image.depth());
    const U last_row = static_cast<U>(image.height() - 1) *
                       static_cast<U>(image.row_pitch());
    return std::make_pair(begin, begin + last_row + row_bytes);
  };
  const auto ra = range(a);
  const auto rb = range(b);
  return ra.first < rb.second && rb.first < ra.second;
}

inline void require_same_layout(const ImageSpec& a, const ImageSpec& b) {
  if (a.width != b.width || a.height != b.height) {
    throw std::invalid_argument("image extents must match");
  }
  if (a.depth != b.depth) {
    throw std::invalid_argument("image pixel depths must match");
  }
  if (a.channels != b.channels) {
    throw std::invalid_argument("image channel counts must match");
  }
}

inline void require_integer(const ImageSpec& image) {
  if (!is_integer(image.depth)) {
    throw std::invalid_argument("operation requires an integer pixel depth");
  }
}

/*
 * An operation mask is single-channel 8-bit, as in OpenCV.
 *
 * Mask semantics are "do not write", not "write zero": where the mask is zero
 * the destination keeps its prior contents, so a masked call must pass an
 * initialized dst. Kernels get this by folding the mask into the store
 * predicate. See docs/operator-development.md.
 */
inline void require_mask(const ImageSpec& mask, const ImageSpec& dst) {
  if (mask.width != dst.width || mask.height != dst.height) {
    throw std::invalid_argument("mask extent must match the destination");
  }
  if (mask.channels != 1 || mask.depth != Depth::k8U) {
    throw std::invalid_argument("mask must be 8-bit single-channel");
  }
}

}  // namespace flagcv

#endif  // FLAGCV_SRC_LOWERING_HELPERS_HPP_
