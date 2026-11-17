"""Triton permutation for the OpenCV-compatible single-channel flip profile."""

import triton
import triton.language as tl


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
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    mode = (flip_code > 0) - (flip_code < 0)
    block = 1024
    count = src.numel()
    _flip_kernel[(triton.cdiv(count, block),)](
        src, dst, count, height, WIDTH=width, MODE=mode, BLOCK=block, num_warps=4
    )
    return dst
