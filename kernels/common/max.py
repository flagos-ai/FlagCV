"""Elementwise two-matrix maximum with OpenCV CUDA ``Max`` functor semantics.

``dst[i] = max(src1[i], src2[i])`` over the flat element space (all channels /
dims are independent).  ``dst`` is preallocated by the caller and returned
in place.  One Triton kernel launch per call; no host copies.
"""

import torch
import triton
import triton.language as tl

_I64_LIMIT = 0x7FFE0000


def _flat_configs():
    pairs = [
        (128, 1),
        (256, 1),
        (256, 2),
        (512, 2),
        (512, 4),
        (1024, 2),
        (1024, 4),
        (2048, 4),
        (2048, 8),
        (4096, 4),
        (4096, 8),
        (8192, 8),
        (16384, 8),
    ]
    return [triton.Config({"BLOCK": b}, num_warps=w) for b, w in pairs]


@triton.autotune(
    configs=_flat_configs(),
    key=["n_elements", "itemsize"],
    warmup=2,
    rep=5,
)
@triton.jit
def _max_flat_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    n_elements,
    itemsize,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    if base + BLOCK <= n_elements:
        # Whole tile in range: unpredicated (fully vectorized) access.
        tl.store(
            o_ptr + offs,
            tl.maximum(
                tl.load(a_ptr + offs, eviction_policy="evict_first"),
                tl.load(b_ptr + offs, eviction_policy="evict_first"),
            ),
            eviction_policy="evict_first",
        )
    else:
        mask = offs < n_elements
        tl.store(
            o_ptr + offs,
            tl.maximum(
                tl.load(a_ptr + offs, mask=mask, eviction_policy="evict_first"),
                tl.load(b_ptr + offs, mask=mask, eviction_policy="evict_first"),
            ),
            mask=mask,
            eviction_policy="evict_first",
        )


@triton.jit
def _max_strided_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lin = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = lin < n_elements
    rem = lin
    off_a = tl.zeros([BLOCK], dtype=tl.int64)
    off_b = tl.zeros([BLOCK], dtype=tl.int64)
    off_o = tl.zeros([BLOCK], dtype=tl.int64)
    for j in tl.static_range(len(SHAPE) - 1, -1, -1):
        idx = rem % SHAPE[j]
        rem = rem // SHAPE[j]
        off_a += idx * A_STRIDES[j]
        off_b += idx * B_STRIDES[j]
        off_o += idx * OUT_STRIDES[j]
    a = tl.load(a_ptr + off_a, mask=mask, eviction_policy="evict_first")
    b = tl.load(b_ptr + off_b, mask=mask, eviction_policy="evict_first")
    tl.store(
        o_ptr + off_o,
        tl.maximum(a, b),
        mask=mask,
        eviction_policy="evict_first",
    )


def _collapse_layout(tensors):
    """Drop size-1 dims and merge adjacent dims that are contiguous in all tensors."""
    shape = list(tensors[0].shape)
    strides = [list(t.stride()) for t in tensors]

    i = 0
    while i < len(shape):
        if shape[i] == 1:
            del shape[i]
            for s in strides:
                del s[i]
        else:
            i += 1

    i = 0
    while i < len(shape) - 1:
        if all(s[i] == s[i + 1] * shape[i + 1] for s in strides):
            shape[i] *= shape[i + 1]
            del shape[i + 1]
            for s in strides:
                s[i] = s.pop(i + 1)
        else:
            i += 1

    if not shape:
        shape = [1]
        strides = [[0] for _ in strides]
    return shape, strides


def max(src1, src2, dst=None):
    if src1.ndim not in (2, 3):
        raise ValueError("Expected a 2-D or HWC image")
    if (src2.shape, src2.dtype, src2.device) != (src1.shape, src1.dtype, src1.device):
        raise ValueError("Sources must match in shape, dtype, and device")
    if dst is not None and (dst.shape, dst.dtype, dst.device) != (
        src1.shape,
        src1.dtype,
        src1.device,
    ):
        raise ValueError("Destination must match sources in shape, dtype, and device")
    with torch.cuda.device(src1.device):
        return _run(src1, src2, dst)


def _run(src1, src2, dst=None):
    if dst is None:
        dst = torch.empty_like(src1)

    n = int(src1.numel())
    if n == 0:
        return dst

    if src1.is_contiguous() and src2.is_contiguous() and dst.is_contiguous():
        _max_flat_kernel[lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)](
            src1,
            src2,
            dst,
            n,
            int(src1.element_size()),
            I64=n > _I64_LIMIT,
        )
        return dst

    shape, strides = _collapse_layout((src1, src2, dst))
    sblock = 1024
    grid = ((n + sblock - 1) // sblock,)
    _max_strided_kernel[grid](
        src1,
        src2,
        dst,
        n,
        SHAPE=tuple(shape),
        A_STRIDES=tuple(strides[0]),
        B_STRIDES=tuple(strides[1]),
        OUT_STRIDES=tuple(strides[2]),
        BLOCK=sblock,
        num_warps=8,
    )
    return dst
