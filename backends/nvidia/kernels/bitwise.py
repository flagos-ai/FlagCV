# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA bitwise kernels using 32-bit packed access.

Bitwise operations are purely bandwidth-bound, so the only thing that matters
is how wide each lane's access is. The common kernel moves one component per
lane; for 8-bit images that wastes three quarters of the achievable throughput
because a byte load occupies a full lane slot.

This variant reinterprets four consecutive 8-bit components as one int32 and
applies the operation on the packed word. Bitwise operators are per-bit, so
packing is exact rather than an approximation: no carry or rounding crosses a
component boundary. The same trick does not generalize to arithmetic kernels.

The packed path needs each row to start 4-byte aligned and to hold a multiple
of four components. The dispatch layer checks both and falls back to the common
kernel otherwise, so this file only handles the packed case.

Mask handling is unchanged: the mask is folded into the store predicate so
masked-out pixels keep their prior contents. A packed word covers four
components, so it may straddle pixels with differing mask values; the kernel
therefore builds a per-byte blend of the packed result and the prior
destination word rather than predicating the whole word.
"""

import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _apply_packed(x, y, MODE: tl.constexpr):
    if MODE == 0:
        return x & y
    elif MODE == 1:
        return x | y
    else:
        return x ^ y


@triton.jit
def bitwise_binary_packed_kernel(
    src1_pointer,
    src2_pointer,
    mask_pointer,
    destination_pointer,
    width_in_words,
    height,
    channels,
    src1_row_words,
    src2_row_words,
    mask_row_pitch,
    destination_row_words,
    MODE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Applies a bitwise binary operation over int32-packed 8-bit rows.

    Pointers are typed int32 by the launcher, so ``width_in_words`` and the row
    strides are all counted in 32-bit words rather than bytes.
    """
    x_offsets = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    y_offsets = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)

    in_range = (y_offsets < height)[:, None] & (x_offsets < width_in_words)[
        None, :
    ]

    src1_offsets = y_offsets[:, None] * src1_row_words + x_offsets[None, :]
    src2_offsets = y_offsets[:, None] * src2_row_words + x_offsets[None, :]
    destination_offsets = (
        y_offsets[:, None] * destination_row_words + x_offsets[None, :]
    )

    left = tl.load(src1_pointer + src1_offsets, mask=in_range, other=0)
    right = tl.load(src2_pointer + src2_offsets, mask=in_range, other=0)
    result = _apply_packed(left, right, MODE)

    if HAS_MASK:
        # One packed word spans four components, which may belong to pixels
        # with different mask values. Blend per byte against the destination's
        # prior contents instead of predicating the whole word.
        prior = tl.load(
            destination_pointer + destination_offsets, mask=in_range, other=0
        )
        keep = tl.zeros_like(result)
        for byte in tl.static_range(4):
            component = x_offsets * 4 + byte
            mask_x = component // channels
            mask_offsets = (
                y_offsets[:, None] * mask_row_pitch + mask_x[None, :]
            )
            mask_values = tl.load(
                mask_pointer + mask_offsets, mask=in_range, other=0
            )
            # Selected bytes take the result; the rest take the prior word.
            selected = (mask_values != 0).to(tl.int32)
            keep = keep | (selected << (byte * 8))
        byte_mask = (keep & 0x01010101) * 0xFF
        result = (result & byte_mask) | (prior & ~byte_mask)

    tl.store(
        destination_pointer + destination_offsets, result, mask=in_range
    )


@triton.jit
def bitwise_not_packed_kernel(
    src_pointer,
    mask_pointer,
    destination_pointer,
    width_in_words,
    height,
    channels,
    src_row_words,
    mask_row_pitch,
    destination_row_words,
    HAS_MASK: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    x_offsets = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    y_offsets = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)

    in_range = (y_offsets < height)[:, None] & (x_offsets < width_in_words)[
        None, :
    ]

    src_offsets = y_offsets[:, None] * src_row_words + x_offsets[None, :]
    destination_offsets = (
        y_offsets[:, None] * destination_row_words + x_offsets[None, :]
    )

    value = tl.load(src_pointer + src_offsets, mask=in_range, other=0)
    result = ~value

    if HAS_MASK:
        prior = tl.load(
            destination_pointer + destination_offsets, mask=in_range, other=0
        )
        keep = tl.zeros_like(result)
        for byte in tl.static_range(4):
            component = x_offsets * 4 + byte
            mask_x = component // channels
            mask_offsets = (
                y_offsets[:, None] * mask_row_pitch + mask_x[None, :]
            )
            mask_values = tl.load(
                mask_pointer + mask_offsets, mask=in_range, other=0
            )
            selected = (mask_values != 0).to(tl.int32)
            keep = keep | (selected << (byte * 8))
        byte_mask = (keep & 0x01010101) * 0xFF
        result = (result & byte_mask) | (prior & ~byte_mask)

    tl.store(
        destination_pointer + destination_offsets, result, mask=in_range
    )
