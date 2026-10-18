/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_PIPELINE_LOWERING_LOWERING_HPP_
#define FLAGCV_SRC_PIPELINE_LOWERING_LOWERING_HPP_

#include "src/image/types.hpp"
#include "src/pipeline/types.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace flagcv {

/*
 * Result of lowering one operation. Parameters are plain integers consumed by
 * the external compiler; lowering never decides block sizes, shared memory or
 * other kernel capabilities, which belong to the platform compiler.
 */
struct LoweredOperation {
  /* Registry key naming the kernel family to resolve. */
  std::string kernel_operation;
  /* Named image operands in kernel argument order. */
  std::vector<std::pair<std::string, ImageSpec>> images;
  /* Device-resident scalar operands, given as an element count. */
  std::vector<std::pair<std::string, std::int32_t>> scalar_operands;
  std::vector<std::pair<std::string, std::int64_t>> parameters;

  void add_image(std::string name, const ImageSpec& image) {
    images.emplace_back(std::move(name), image);
  }

  void add_scalar_operand(std::string name, std::int32_t element_count) {
    scalar_operands.emplace_back(std::move(name), element_count);
  }

  void add_parameter(std::string name, std::int64_t value) {
    parameters.emplace_back(std::move(name), value);
  }
};

/* Single dispatch point mapping an operation to its lowering function. */
LoweredOperation lower_operation(const OperationSpec& operation);

LoweredOperation lower_bitwise(const OperationSpec& operation);

}  // namespace flagcv

#endif  // FLAGCV_SRC_PIPELINE_LOWERING_LOWERING_HPP_
