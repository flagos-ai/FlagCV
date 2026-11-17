"""Triton candidate for the unmasked single-channel uint8 complement profile."""

import triton
import triton.language as tl


@triton.jit
def _bitwise_not_kernel(src, dst, count, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        values = tl.load(src + offsets)
        tl.store(dst + offsets, ~values)
    else:
        valid = offsets < count
        values = tl.load(src + offsets, valid)
        tl.store(dst + offsets, ~values, valid)


def bitwise_not(src, dst):
    count = src.numel()
    block = 2048
    _bitwise_not_kernel[(triton.cdiv(count, block),)](
        src, dst, count, BLOCK=block, EVEN=count % block == 0, num_warps=4
    )
    return dst
