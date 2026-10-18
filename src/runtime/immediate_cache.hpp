/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_RUNTIME_IMMEDIATE_CACHE_HPP_
#define FLAGCV_SRC_RUNTIME_IMMEDIATE_CACHE_HPP_

#include <flagcv/flagcv.hpp>

#include "src/pipeline/types.hpp"

#include <vector>

namespace flagcv {

/*
 * Runs a single operation through the pipeline path.
 *
 * The built executable is cached per (operation, operand layout, attributes),
 * so repeated immediate calls with the same layout pay build cost only once.
 * This is what keeps the immediate surface from being a second implementation:
 * it goes through the same lowering, the same registry and the same kernels.
 *
 * Work is enqueued on `stream` and is not synchronized.
 */
void execute_immediate(Handle& handle,
                       const OperationSpec& operation,
                       const std::vector<flagcvBinding_t>& bindings,
                       flagcvStream_t stream);

}  // namespace flagcv

#endif  // FLAGCV_SRC_RUNTIME_IMMEDIATE_CACHE_HPP_
