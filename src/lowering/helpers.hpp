/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_LOWERING_HELPERS_HPP_
#define FLAGCV_SRC_LOWERING_HELPERS_HPP_

#include <flagcv/gpumat.hpp>

#include "src/image/types.hpp"

#include <stdexcept>
#include <string>

namespace flagcv {

/* Builds the internal spec, converting the byte pitch to an element stride. */
inline ImageSpec to_spec(const GpuMat& image, const std::string& name) {
  if (image.empty()) {
    throw std::invalid_argument("operand '" + name + "' must not be empty");
  }
  ImageSpec spec;
  spec.name = name;
  spec.width = image.width();
  spec.height = image.height();
  spec.depth = image.depth();
  spec.channels = image.channels();
  spec.row_stride = image.row_stride();
  return spec;
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
