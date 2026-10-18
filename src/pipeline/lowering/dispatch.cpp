/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/pipeline/lowering/lowering.hpp"

#include "src/pipeline/lowering/helpers.hpp"

#include <stdexcept>

namespace flagcv {
namespace {

bool is_bitwise_mode(flagcvElementwiseMode_t mode) {
  switch (mode) {
    case FLAGCV_ELEMENTWISE_BITWISE_AND:
    case FLAGCV_ELEMENTWISE_BITWISE_OR:
    case FLAGCV_ELEMENTWISE_BITWISE_XOR:
    case FLAGCV_ELEMENTWISE_BITWISE_NOT:
      return true;
    default:
      return false;
  }
}

}  // namespace

/*
 * Single dispatch point from operation to kernel family. Adding an operator
 * family means adding one lowering file and one branch here; no other shared
 * file changes.
 */
LoweredOperation lower_operation(const OperationSpec& operation) {
  if (!operation.finalized) {
    throw std::invalid_argument(
        "operation must be finalized before lowering");
  }

  switch (operation.operation) {
    case FLAGCV_OPERATION_ELEMENTWISE: {
      const auto mode = static_cast<flagcvElementwiseMode_t>(
          require_int64(operation, "mode"));
      if (is_bitwise_mode(mode)) {
        return lower_bitwise(operation);
      }
      throw std::invalid_argument("elementwise mode is not implemented yet");
    }
    case FLAGCV_OPERATION_CVT_COLOR:
    case FLAGCV_OPERATION_RESIZE:
    case FLAGCV_OPERATION_FILTER:
    case FLAGCV_OPERATION_MORPHOLOGY:
    case FLAGCV_OPERATION_THRESHOLD:
    case FLAGCV_OPERATION_CUSTOM:
      throw std::invalid_argument("operation is not implemented yet");
  }
  throw std::invalid_argument("unknown operation");
}

}  // namespace flagcv
