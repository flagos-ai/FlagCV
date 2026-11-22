"""Triton implementation of two-matrix elementwise minimum (OpenCV CUDA style).

dst[i] = min(src1[i], src2[i]) for every linear element index.

Pure streaming compare+select: three operand streams, no reuse.
Floating semantics are fmin-style (a NaN operand is discarded), which is what
the reference produces for special values.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------
# torch "barebones" unsigned / bool dtypes are re-interpreted (zero-copy view)
# as the signed integer of the same width so the kernel only sees signed
# pointers; unsigned ordering is restored arithmetically in the kernel body.
_SIGNED_VIEW = {
    torch.uint8: torch.int8,
    torch.uint16: torch.int16,
    torch.uint32: torch.int32,
    torch.uint64: torch.int64,
    torch.bool: torch.uint8,
}

_UNSIGNED_DTYPES = (torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.bool)


@triton.jit
def _min_combine(
    a, b, IS_FLOAT: tl.constexpr, IS_UNSIGNED: tl.constexpr, BITS: tl.constexpr
):
    """Bit-exact min of two equally typed operands (single return point)."""
    if IS_FLOAT:
        # fmin semantics: NaN operand is discarded, signed zeros handled natively.
        out = tl.minimum(a, b)
    elif IS_UNSIGNED:
        if BITS == 8:
            m = tl.minimum(a.to(tl.int32) & 255, b.to(tl.int32) & 255)
            out = tl.where(m >= 128, m - 256, m).to(tl.int8)
        elif BITS == 16:
            m = tl.minimum(a.to(tl.int32) & 65535, b.to(tl.int32) & 65535)
            out = tl.where(m >= 32768, m - 65536, m).to(tl.int16)
        elif BITS == 32:
            flip = -2147483648
            out = tl.minimum(a ^ flip, b ^ flip) ^ flip
        else:
            flip = -9223372036854775808
            out = tl.minimum(a ^ flip, b ^ flip) ^ flip
    else:
        out = tl.minimum(a, b)
    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=1),
        triton.Config({"BLOCK": 512}, num_warps=2),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=4),
        triton.Config({"BLOCK": 8192}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _min_flat_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    n_elements,
    IS_FLOAT: tl.constexpr,
    IS_UNSIGNED: tl.constexpr,
    BITS: tl.constexpr,
    NEED_I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if NEED_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask, cache_modifier=".cg")
    b = tl.load(b_ptr + offs, mask=mask, cache_modifier=".cg")
    out = _min_combine(a, b, IS_FLOAT, IS_UNSIGNED, BITS)
    tl.store(o_ptr + offs, out, mask=mask)


@triton.jit
def _min_strided4_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    n_elements,
    s1,
    s2,
    s3,
    s4,
    a1,
    a2,
    a3,
    a4,
    b1,
    b2,
    b3,
    b4,
    o1,
    o2,
    o3,
    o4,
    IS_FLOAT: tl.constexpr,
    IS_UNSIGNED: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    idx = offs
    i4 = idx % s4
    idx = idx // s4
    i3 = idx % s3
    idx = idx // s3
    i2 = idx % s2
    i1 = idx // s2
    aoff = i1 * a1 + i2 * a2 + i3 * a3 + i4 * a4
    boff = i1 * b1 + i2 * b2 + i3 * b3 + i4 * b4
    ooff = i1 * o1 + i2 * o2 + i3 * o3 + i4 * o4
    a = tl.load(a_ptr + aoff, mask=mask)
    b = tl.load(b_ptr + boff, mask=mask)
    out = _min_combine(a, b, IS_FLOAT, IS_UNSIGNED, BITS)
    tl.store(o_ptr + ooff, out, mask=mask)


# ---------------------------------------------------------------------------
# host dispatch
# ---------------------------------------------------------------------------
_BLOCK = 1024
_WARPS = 4


def _merge_dims(shape, strides):
    """Squeeze unit dims and merge adjacent dims consistent for every operand."""
    dims = []
    for i in range(len(shape)):
        if shape[i] == 1:
            continue
        dims.append([shape[i]] + [st[i] for st in strides])
    if not dims:
        return [1], [[0] * len(strides)]

    merged = [dims[-1]]
    for k in range(len(dims) - 2, -1, -1):
        cur = dims[k]
        nxt = merged[0]
        ok = True
        for j in range(len(strides)):
            if cur[1 + j] != nxt[1 + j] * nxt[0]:
                ok = False
                break
        if ok:
            merged[0] = [cur[0] * nxt[0], *nxt[1:]]
        else:
            merged.insert(0, cur)
    out_shape = [d[0] for d in merged]
    out_strides = [[d[1 + j] for d in merged] for j in range(len(strides))]
    return out_shape, out_strides


def min(src1, src2, dst=None):
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

    n = src1.numel()
    if n == 0:
        return dst

    dt = src1.dtype
    unsigned = dt in _UNSIGNED_DTYPES
    is_float = dt.is_floating_point
    bits = dt.itemsize * 8
    need_i64 = n > 0x7FFFFFFF

    a = src1
    b = src2
    o = dst
    view_dtype = _SIGNED_VIEW.get(dt)
    if view_dtype is not None:
        a = a.view(view_dtype)
        b = b.view(view_dtype)
        o = o.view(view_dtype)

    if a.is_contiguous() and b.is_contiguous() and o.is_contiguous():
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        _min_flat_kernel[grid](
            a,
            b,
            o,
            n,
            is_float,
            unsigned,
            bits,
            need_i64,
        )
        return dst

    shape = list(a.shape)
    if len(shape) == 0:
        shape = [1]
        a = a.view(shape)
        b = b.view(shape)
        o = o.view(shape)
    mshape, (ma, mb, mo) = _merge_dims(shape, [a.stride(), b.stride(), o.stride()])
    if len(mshape) > 4:
        raise NotImplementedError("unsupported operand layout for elementwise min")
    while len(mshape) < 4:
        mshape.insert(0, 1)
        ma.insert(0, 0)
        mb.insert(0, 0)
        mo.insert(0, 0)
    grid = (triton.cdiv(n, _BLOCK),)
    _min_strided4_kernel[grid](
        a,
        b,
        o,
        n,
        mshape[0],
        mshape[1],
        mshape[2],
        mshape[3],
        ma[0],
        ma[1],
        ma[2],
        ma[3],
        mb[0],
        mb[1],
        mb[2],
        mb[3],
        mo[0],
        mo[1],
        mo[2],
        mo[3],
        is_float,
        unsigned,
        bits,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )
    return dst
