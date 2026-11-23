"""Triton candidate for the unmasked single-channel uint8 AND profile."""

import triton
import triton.language as tl


@triton.jit
def _bitwise_and_kernel(src1, src2, dst, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < count
    left = tl.load(src1 + offsets, valid)
    right = tl.load(src2 + offsets, valid)
    tl.store(dst + offsets, left & right, valid)


def bitwise_and(src1, src2, dst):
    count = src1.numel()
    block = 1024
    _bitwise_and_kernel[(triton.cdiv(count, block),)](
        src1, src2, dst, count, BLOCK=block, num_warps=4
    )
    return dst
