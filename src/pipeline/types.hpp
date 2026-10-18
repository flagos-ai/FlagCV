/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_PIPELINE_TYPES_HPP_
#define FLAGCV_SRC_PIPELINE_TYPES_HPP_

#include <flagcv/flagcv.h>

#include "src/image/types.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace flagcv {

using AttributeValue = std::variant<std::int64_t,
                                    double,
                                    bool,
                                    std::string,
                                    std::vector<std::int64_t>>;
using AttributeMap = std::map<std::string, AttributeValue>;

struct OperationPort {
  std::string name;
  ImageSpec image;
  bool is_optional = false;
};

struct OperationSpec {
  flagcvOperation_t operation = FLAGCV_OPERATION_ELEMENTWISE;
  std::string name;
  std::vector<OperationPort> inputs;
  std::vector<OperationPort> outputs;
  AttributeMap attributes;
  bool finalized = false;
};

struct PipelineSpec {
  std::string name;
  std::vector<OperationSpec> operations;
  bool finalized = false;
};

}  // namespace flagcv

#endif  // FLAGCV_SRC_PIPELINE_TYPES_HPP_
