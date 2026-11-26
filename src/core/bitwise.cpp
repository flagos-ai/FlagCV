/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/core.hpp>

#include "src/lowering/lowering.hpp"
#include "src/runtime/runtime.hpp"

namespace flagcv {

void bitwise_and(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                 const GpuMat& dst, Stream stream) {
  const LoweredOp op = lower_pointwise(src1, src2, dst, PointwiseMode::kAnd);
  launch(handle, op, {src1.data(), src2.data(), dst.data()}, stream);
}

void bitwise_not(Handle& handle, const GpuMat& src, const GpuMat& dst,
                 Stream stream) {
  const LoweredOp op = lower_pointwise(src, src, dst, PointwiseMode::kNot);
  launch(handle, op, {src.data(), src.data(), dst.data()}, stream);
}

}  // namespace flagcv
