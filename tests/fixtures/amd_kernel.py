# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Test-only kernel; no operator is registered in the production library."""

import triton
import triton.language as tl


@triton.jit
def masked_xor(
    src,
    dst,
    mask,
    width,
    height,
    channels,
    src_stride,
    dst_stride,
    mask_stride,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index // (width * channels)
    col = index % (width * channels)
    valid = row < height
    enabled = (
        tl.load(mask + row * mask_stride + col // channels, mask=valid, other=0) != 0
    )
    value = tl.load(src + row * src_stride + col, mask=valid, other=0)
    tl.store(dst + row * dst_stride + col, value ^ 0x5A, mask=valid & enabled)
