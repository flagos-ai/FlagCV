"""Triton subtraction with OpenCV uint8 saturation on underflow."""

import torch
import triton
import triton.language as tl


@triton.jit
def _subtract_kernel(a, b, dst, count, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = True if EVEN else offsets < count
    left = tl.load(a + offsets, valid)
    right = tl.load(b + offsets, valid)
    if a.dtype.element_ty == tl.uint8:
        difference = tl.maximum(left.to(tl.int32) - right.to(tl.int32), 0)
    else:
        difference = left - right
    tl.store(dst + offsets, difference, valid)


def subtract(a, b, dst):
    count = a.numel()
    block = 2048 if a.dtype == torch.uint8 else 1024
    _subtract_kernel[(triton.cdiv(count, block),)](
        a, b, dst, count, BLOCK=block, EVEN=count % block == 0, num_warps=4
    )
    return dst
