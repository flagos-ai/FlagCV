"""Triton multiplication at unit scale with OpenCV uint8 saturation."""

import triton
import triton.language as tl


@triton.jit
def _multiply_kernel(a, b, dst, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < count
    left = tl.load(a + offsets, valid)
    right = tl.load(b + offsets, valid)
    if a.dtype.element_ty == tl.uint8:
        product = tl.minimum(left.to(tl.int32) * right.to(tl.int32), 255)
    else:
        product = left * right
    tl.store(dst + offsets, product, valid)


def multiply(a, b, dst):
    count = a.numel()
    block = 1024
    _multiply_kernel[(triton.cdiv(count, block),)](
        a, b, dst, count, BLOCK=block, num_warps=4
    )
    return dst
