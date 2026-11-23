"""FlagCV bitwise_or: fused single-kernel Triton implementation.

Semantics (OpenCV CUDA element_operations.cpp two-matrix bitwise_or):
  unmasked: dst[i] = raw(src1[i]) | raw(src2[i])            for all i < N
  masked  : dst[i] = raw(src1[i]) | raw(src2[i])            where mask[i // CPG] != 0
                     dst[i] untouched (preserved)            otherwise

float32 operands are handled on their raw 32-bit representation (bitcast), so the
result is bit-identical to the reference. dst is written in place and returned.
"""

import torch
import triton
import triton.language as tl

_MAX_RANK = 4

# Bounded configuration set for the flat kernel. The benchmarked shapes are small
# (3.9e3 .. 7.9e5 elements), so elements-per-thread and the number of resident
# blocks both matter; the tuner picks per (n_elements, dtype, HAS_MASK) key.
_CONFIGS = [
    triton.Config({"BLOCK": 128}, num_warps=1),
    triton.Config({"BLOCK": 256}, num_warps=1),
    triton.Config({"BLOCK": 512}, num_warps=1),
    triton.Config({"BLOCK": 512}, num_warps=2),
    triton.Config({"BLOCK": 1024}, num_warps=2),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 8192}, num_warps=4),
    triton.Config({"BLOCK": 8192}, num_warps=8),
]

# dtype -> same-itemsize integer dtype used for the bit-exact raw-bit path
_VIEW_DTYPE = {
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float64: torch.int64,
    torch.bool: torch.uint8,
}
for _name, _target in (
    ("uint16", torch.int16),
    ("uint32", torch.int32),
    ("uint64", torch.int64),
):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _VIEW_DTYPE[_dt] = _target


@triton.autotune(configs=_CONFIGS, key=["n_elements", "HAS_MASK"], warmup=5, rep=20)
@triton.jit
def _bitwise_or_flat(
    s1_ptr,
    s2_ptr,
    dst_ptr,
    m_ptr,
    n_elements,
    CPG: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    guard = offs < n_elements

    a = tl.load(s1_ptr + offs, mask=guard, other=0)
    b = tl.load(s2_ptr + offs, mask=guard, other=0)
    out = (a | b).to(dst_ptr.dtype.element_ty)

    if HAS_MASK:
        mv = tl.load(m_ptr + offs // CPG, mask=guard, other=0)
        tl.store(dst_ptr + offs, out, mask=guard & (mv != 0))
    else:
        tl.store(dst_ptr + offs, out, mask=guard)


@triton.jit
def _decompose(idx, D1, D2, D3, s0, s1, s2, s3):
    i3 = idx % D3
    idx = idx // D3
    i2 = idx % D2
    idx = idx // D2
    i1 = idx % D1
    i0 = idx // D1
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3


@triton.jit
def _bitwise_or_strided(
    s1_ptr,
    s2_ptr,
    dst_ptr,
    m_ptr,
    n_elements,
    D1,
    D2,
    D3,
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    b2,
    b3,
    c0,
    c1,
    c2,
    c3,
    E1,
    E2,
    E3,
    e0,
    e1,
    e2,
    e3,
    CPG: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    guard = idx < n_elements

    o1 = _decompose(idx, D1, D2, D3, a0, a1, a2, a3)
    o2 = _decompose(idx, D1, D2, D3, b0, b1, b2, b3)
    od = _decompose(idx, D1, D2, D3, c0, c1, c2, c3)

    va = tl.load(s1_ptr + o1, mask=guard, other=0)
    vb = tl.load(s2_ptr + o2, mask=guard, other=0)
    out = (va | vb).to(dst_ptr.dtype.element_ty)

    if HAS_MASK:
        mi = idx // CPG
        om = _decompose(mi, E1, E2, E3, e0, e1, e2, e3)
        mv = tl.load(m_ptr + om, mask=guard, other=0)
        tl.store(dst_ptr + od, out, mask=guard & (mv != 0))
    else:
        tl.store(dst_ptr + od, out, mask=guard)


def _int_view(t):
    """Zero-copy same-itemsize integer view so bitwise ops act on raw bits."""
    target = _VIEW_DTYPE.get(t.dtype)
    if target is None:
        return t
    try:
        return t.view(target)
    except Exception:
        return t


def _flat(t):
    if t.dim() == 0:
        return t.view(1)
    return t.view(-1)


def _shape_stride4(t):
    shape = list(t.shape)
    strides = list(t.stride())
    while len(shape) < _MAX_RANK:
        shape.insert(0, 1)
        strides.insert(0, 0)
    return shape, strides


def _config(itemsize):
    if itemsize == 1:
        return 4096, 4
    if itemsize == 2:
        return 2048, 4
    return 1024, 4


def bitwise_or(src1, src2, dst=None, mask=None):
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
    if mask is not None:
        if src1.ndim == 3 and src1.shape[-1] != 1:
            raise ValueError("Mask requires single-channel inputs")
        if (mask.shape, mask.dtype, mask.device) != (
            src1.shape[:2],
            torch.uint8,
            src1.device,
        ):
            raise ValueError("Mask must be an HxW uint8 tensor on the input device")
    with torch.cuda.device(src1.device):
        return _run(src1, src2, dst, mask)


def _run(src1, src2, dst=None, mask=None):
    if dst is None:
        dst = torch.empty_like(src1)

    s1 = _int_view(src1)
    s2 = _int_view(src2)
    d = _int_view(dst)

    n = d.numel()
    if n == 0:
        return dst

    block, warps = _config(d.element_size())
    grid = (triton.cdiv(n, block),)

    has_mask = 0
    cpg = 1
    mv = None
    if mask is not None:
        has_mask = 1
        mv = mask
        if mv.dtype == torch.bool:
            mv = mv.view(torch.uint8)
        mn = mv.numel()
        if mn > 0 and n % mn == 0 and mn <= n:
            cpg = n // mn
        if cpg < 1:
            cpg = 1

    if not has_mask:
        mv = s1  # dummy pointer, never dereferenced

    contiguous = s1.is_contiguous() and s2.is_contiguous() and d.is_contiguous()
    if has_mask and not mv.is_contiguous():
        contiguous = False

    if contiguous:
        f1 = _flat(s1)
        f2 = _flat(s2)
        fd = _flat(d)
        fm = _flat(mv) if has_mask else f1
        _bitwise_or_flat[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
            f1,
            f2,
            fd,
            fm,
            n,
            CPG=cpg,
            HAS_MASK=has_mask,
        )
        return dst

    # generic strided path (non-contiguous operands, still zero-copy)
    dshape, _ = _shape_stride4(d)
    _, a = _shape_stride4(s1)
    _, b = _shape_stride4(s2)
    _, c = _shape_stride4(d)
    if has_mask:
        mshape, e = _shape_stride4(mv)
    else:
        mshape, e = (1, 1, 1, 1), (0, 0, 0, 0)

    _bitwise_or_strided[grid](
        s1,
        s2,
        d,
        mv,
        n,
        dshape[1],
        dshape[2],
        dshape[3],
        a[0],
        a[1],
        a[2],
        a[3],
        b[0],
        b[1],
        b[2],
        b[3],
        c[0],
        c[1],
        c[2],
        c[3],
        mshape[1],
        mshape[2],
        mshape[3],
        e[0],
        e[1],
        e[2],
        e[3],
        CPG=cpg,
        HAS_MASK=has_mask,
        BLOCK=block,
        num_warps=warps,
    )
    return dst
