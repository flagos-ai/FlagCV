# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mirror an image around its horizontal axis, vertical axis, or both."""

from operator import index

import triton
import triton.language as tl


@triton.jit
def flip_kernel(
    src_ptr,
    dst_ptr,
    width,
    height,
    src_row_stride,
    dst_row_stride,
    CHANNELS: tl.constexpr,
    FLIP_X: tl.constexpr,
    FLIP_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """dst(y, x, c) = src(y', x', c), y' and x' mirrored when FLIP_Y / FLIP_X.

    The x axis of the grid walks channel values, not pixels: a row holds
    width * CHANNELS elements, so interleaved channels stay contiguous in
    memory and the kernel needs no per-channel loop. Row strides are in
    elements (see docs/architecture.md), and row offsets are widened to int64
    so large padded images cannot overflow.
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    elem = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    row = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    in_bounds = (row[:, None] < height) & (elem[None, :] < width * CHANNELS)

    if FLIP_X:
        pixel = elem // CHANNELS
        src_elem = (width - 1 - pixel) * CHANNELS + (elem - pixel * CHANNELS)
    else:
        src_elem = elem
    if FLIP_Y:
        src_row = height - 1 - row
    else:
        src_row = row

    src_offsets = src_row.to(tl.int64)[:, None] * src_row_stride + src_elem[None, :]
    dst_offsets = row.to(tl.int64)[:, None] * dst_row_stride + elem[None, :]
    values = tl.load(src_ptr + src_offsets, mask=in_bounds)
    tl.store(dst_ptr + dst_offsets, values, mask=in_bounds)


@triton.jit
def _flip_kernel(
    src,
    dst,
    count,
    height,
    WIDTH: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // WIDTH
    col = offsets % WIDTH
    if MODE <= 0:
        row = height - 1 - row
    if MODE != 0:
        col = WIDTH - 1 - col
    valid = offsets < count
    values = tl.load(src + row * WIDTH + col, valid)
    tl.store(dst + offsets, values, valid)


def flip(src, dst, flip_code):
    height, width = src.shape
    flip_code = index(flip_code)
    mode = (flip_code > 0) - (flip_code < 0)
    block = 1024
    count = src.numel()
    _flip_kernel[(triton.cdiv(count, block),)](
        src, dst, count, height, WIDTH=width, MODE=mode, BLOCK=block, num_warps=4
    )
    return dst
