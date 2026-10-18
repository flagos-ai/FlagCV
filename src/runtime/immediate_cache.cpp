/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/runtime/immediate_cache.hpp"

#include "src/pipeline/lowering/lowering.hpp"
#include "src/runtime/kernel_resolution.hpp"

namespace flagcv {

/*
 * TODO: implement the build-and-cache path.
 *
 * Steps, in order:
 *   1. Lower the operation, which also validates the operands.
 *   2. Hash (kernel_operation, image layouts, parameters) into a cache key.
 *   3. On a miss: resolve the kernel through resolve_kernel(), which consults
 *      the platform registry before the common one; serialize the lowered form
 *      to IR; run the external compiler; build a backend executable.
 *   4. Execute with the caller's bindings on the caller's stream.
 *
 * Lowering runs here rather than inside the cache lookup so that invalid
 * operands are rejected on every call, not only on a cache miss.
 */
void execute_immediate(Handle& handle,
                       const OperationSpec& operation,
                       const std::vector<flagcvBinding_t>& bindings,
                       flagcvStream_t stream) {
  const LoweredOperation lowered = lower_operation(operation);

  (void)handle;
  (void)bindings;
  (void)stream;
  (void)lowered;
  throw Error(FLAGCV_STATUS_NOT_SUPPORTED,
              "flagcv: immediate execution path is not implemented yet");
}

}  // namespace flagcv
