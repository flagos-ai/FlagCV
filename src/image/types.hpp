/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_IMAGE_TYPES_HPP_
#define FLAGCV_SRC_IMAGE_TYPES_HPP_

#include <flagcv/types.hpp>

#include <cstdint>
#include <string>

namespace flagcv {

/*
 * Internal form of one image operand.
 *
 * `row_stride` is in ELEMENTS, converted once from the public byte pitch,
 * because kernel pointers are typed by pixel depth and advance in elements.
 */
struct ImageSpec {
  std::string name;
  std::int32_t width = 0;
  std::int32_t height = 0;
  Depth depth = Depth::k8U;
  std::int32_t channels = 0;
  std::int64_t row_stride = 0;
};

}  // namespace flagcv

#endif  // FLAGCV_SRC_IMAGE_TYPES_HPP_
