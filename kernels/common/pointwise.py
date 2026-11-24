# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pitched, interleaved array-array kernels used by the C++ runtime."""

import triton
import triton.language as tl


@triton.jit
def pointwise_kernel(
    src1_ptr,
    src2_ptr,
    dst_ptr,
    width,
    height,
    src1_row_stride,
    src2_row_stride,
    dst_row_stride,
    CHANNELS: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    y = (tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)).to(tl.int64)
    valid = (y[:, None] < height) & (x[None, :] < width * CHANNELS)
    a = tl.load(src1_ptr + y[:, None] * src1_row_stride + x[None, :], valid, 0)
    if MODE != 4:
        b = tl.load(src2_ptr + y[:, None] * src2_row_stride + x[None, :], valid, 0)
    if MODE <= 2:
        if src1_ptr.dtype.element_ty == tl.uint8:
            a = a.to(tl.int32)
            b = b.to(tl.int32)
        if MODE == 0:
            result = a + b
        elif MODE == 1:
            result = a - b
        else:
            result = a * b
        if src1_ptr.dtype.element_ty == tl.uint8:
            result = tl.minimum(tl.maximum(result, 0), 255)
    elif MODE == 3:
        result = a & b
    else:
        result = ~a
    tl.store(dst_ptr + y[:, None] * dst_row_stride + x[None, :], result, valid)
