/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/core.hpp>

#include "src/lowering/lowering.hpp"
#include "src/runtime/runtime.hpp"

namespace flagcv {

void flip(Handle& handle, const GpuMat& src, const GpuMat& dst, int flip_code,
          Stream stream) {
  const LoweredOp op = lower_flip(src, dst, flip_code);
  launch(handle, op, {src.data(), dst.data()}, stream);
}

}  // namespace flagcv
