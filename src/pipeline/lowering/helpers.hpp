/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_PIPELINE_LOWERING_HELPERS_HPP_
#define FLAGCV_SRC_PIPELINE_LOWERING_HELPERS_HPP_

#include "src/pipeline/types.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>

namespace flagcv {

inline const OperationPort* find_port(
    const std::vector<OperationPort>& ports, std::string_view name) {
  const auto found = std::find_if(
      ports.begin(), ports.end(),
      [name](const OperationPort& port) { return port.name == name; });
  return found != ports.end() ? &*found : nullptr;
}

inline bool has_input(const OperationSpec& operation, std::string_view name) {
  return find_port(operation.inputs, name) != nullptr;
}

inline const ImageSpec& require_input(const OperationSpec& operation,
                                      std::string_view name) {
  const OperationPort* port = find_port(operation.inputs, name);
  if (port == nullptr) {
    throw std::invalid_argument("operation is missing input port '" +
                                std::string(name) + "'");
  }
  return port->image;
}

inline const ImageSpec& require_output(const OperationSpec& operation,
                                       std::string_view name) {
  const OperationPort* port = find_port(operation.outputs, name);
  if (port == nullptr) {
    throw std::invalid_argument("operation is missing output port '" +
                                std::string(name) + "'");
  }
  return port->image;
}

inline std::int64_t require_int64(const OperationSpec& operation,
                                  std::string_view name) {
  const auto found = operation.attributes.find(std::string(name));
  if (found == operation.attributes.end()) {
    throw std::invalid_argument("operation is missing attribute '" +
                                std::string(name) + "'");
  }
  const auto* value = std::get_if<std::int64_t>(&found->second);
  if (value == nullptr) {
    throw std::invalid_argument("attribute '" + std::string(name) +
                                "' is not an integer");
  }
  return *value;
}

inline void require_same_extent(const ImageSpec& left,
                                const ImageSpec& right) {
  if (left.width != right.width || left.height != right.height) {
    throw std::invalid_argument("image extents must match");
  }
}

inline void require_same_depth(const ImageSpec& left,
                               const ImageSpec& right) {
  if (left.depth != right.depth || left.channels != right.channels) {
    throw std::invalid_argument("image depth and channel count must match");
  }
}

}  // namespace flagcv

#endif  // FLAGCV_SRC_PIPELINE_LOWERING_HELPERS_HPP_
