# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bitwise elementwise kernels over pitched image planes.

Every load and store is addressed through a byte row pitch, because imaging
allocations commonly pad rows for DMA alignment. Logical extent and storage
extent are therefore independent, and the source and destination pitches are
not assumed to be equal.

The channel axis is folded into the column axis: a pixel's components are
contiguous, so a tile of component slots stays coalesced regardless of channel
count, and the kernel does not need a separate compiled variant per channel
count.

Where the mask is zero the destination is left untouched. The kernel gets this
by folding the mask into the store predicate instead of selecting a value,
which keeps masked-out pixels byte-identical to their prior contents.
"""

import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _apply(x, y, MODE: tl.constexpr):
    if MODE == 0:
        return x & y
    elif MODE == 1:
        return x | y
    else:
        return x ^ y


@triton.jit
def bitwise_binary_kernel(
    src1_pointer,
    src2_pointer,
    mask_pointer,
    destination_pointer,
    width_in_components,
    height,
    channels,
    src1_row_pitch,
    src2_row_pitch,
    mask_row_pitch,
    destination_row_pitch,
    MODE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    x_offsets = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    y_offsets = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)

    in_range = (y_offsets < height)[:, None] & (
        x_offsets < width_in_components
    )[None, :]

    if HAS_MASK:
        # The mask is single-channel, so every component of a pixel consults
        # the same mask element.
        mask_x = x_offsets // channels
        mask_offsets = y_offsets[:, None] * mask_row_pitch + mask_x[None, :]
        mask_values = tl.load(
            mask_pointer + mask_offsets, mask=in_range, other=0
        )
        store_predicate = in_range & (mask_values != 0)
    else:
        store_predicate = in_range

    src1_offsets = y_offsets[:, None] * src1_row_pitch + x_offsets[None, :]
    src2_offsets = y_offsets[:, None] * src2_row_pitch + x_offsets[None, :]
    destination_offsets = (
        y_offsets[:, None] * destination_row_pitch + x_offsets[None, :]
    )

    left = tl.load(src1_pointer + src1_offsets, mask=in_range, other=0)
    right = tl.load(src2_pointer + src2_offsets, mask=in_range, other=0)

    tl.store(
        destination_pointer + destination_offsets,
        _apply(left, right, MODE),
        mask=store_predicate,
    )


@triton.jit
def bitwise_binary_scalar_kernel(
    src_pointer,
    scalar_pointer,
    mask_pointer,
    destination_pointer,
    width_in_components,
    height,
    channels,
    src_row_pitch,
    mask_row_pitch,
    destination_row_pitch,
    MODE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Applies a bitwise operation against a per-channel scalar.

    The scalar arrives as a device buffer of ``channels`` elements so that the
    compiled artifact is not specialized on its value.
    """
    x_offsets = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    y_offsets = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)

    in_range = (y_offsets < height)[:, None] & (
        x_offsets < width_in_components
    )[None, :]

    if HAS_MASK:
        mask_x = x_offsets // channels
        mask_offsets = y_offsets[:, None] * mask_row_pitch + mask_x[None, :]
        mask_values = tl.load(
            mask_pointer + mask_offsets, mask=in_range, other=0
        )
        store_predicate = in_range & (mask_values != 0)
    else:
        store_predicate = in_range

    src_offsets = y_offsets[:, None] * src_row_pitch + x_offsets[None, :]
    destination_offsets = (
        y_offsets[:, None] * destination_row_pitch + x_offsets[None, :]
    )

    value = tl.load(src_pointer + src_offsets, mask=in_range, other=0)
    scalar = tl.load(scalar_pointer + (x_offsets % channels)[None, :])

    tl.store(
        destination_pointer + destination_offsets,
        _apply(value, scalar, MODE),
        mask=store_predicate,
    )


@triton.jit
def bitwise_not_kernel(
    src_pointer,
    mask_pointer,
    destination_pointer,
    width_in_components,
    height,
    channels,
    src_row_pitch,
    mask_row_pitch,
    destination_row_pitch,
    HAS_MASK: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    x_offsets = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    y_offsets = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)

    in_range = (y_offsets < height)[:, None] & (
        x_offsets < width_in_components
    )[None, :]

    if HAS_MASK:
        mask_x = x_offsets // channels
        mask_offsets = y_offsets[:, None] * mask_row_pitch + mask_x[None, :]
        mask_values = tl.load(
            mask_pointer + mask_offsets, mask=in_range, other=0
        )
        store_predicate = in_range & (mask_values != 0)
    else:
        store_predicate = in_range

    src_offsets = y_offsets[:, None] * src_row_pitch + x_offsets[None, :]
    destination_offsets = (
        y_offsets[:, None] * destination_row_pitch + x_offsets[None, :]
    )

    value = tl.load(src_pointer + src_offsets, mask=in_range, other=0)

    tl.store(
        destination_pointer + destination_offsets,
        ~value,
        mask=store_predicate,
    )
