import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Bounded configuration search (per dtype: fp32 and uint8 have very different
# elements-per-thread requirements, so each has its own autotuner / cache).
# ---------------------------------------------------------------------------
F32_CONFIGS = [
    triton.Config({"BLOCK": 128}, num_warps=1),
    triton.Config({"BLOCK": 256}, num_warps=2),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=8),
]
U8_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=1),
    triton.Config({"BLOCK": 512}, num_warps=2),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=8),
]


@triton.autotune(configs=F32_CONFIGS, key=["numel"])
@triton.jit
def _add_f32_kernel(
    a_ptr, b_ptr, d_ptr, numel, alpha, HAS_ALPHA: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    y = tl.load(b_ptr + offs, mask=mask, other=0.0)
    if HAS_ALPHA:
        y = y * alpha
    tl.store(d_ptr + offs, x + y, mask=mask)


@triton.autotune(configs=U8_CONFIGS, key=["numel"])
@triton.jit
def _add_u8_kernel(
    a_ptr, b_ptr, d_ptr, numel, alpha, HAS_ALPHA: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int32)
    y = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int32)
    if HAS_ALPHA:
        y = y * alpha.to(tl.int32)
    s = x + y
    s = tl.minimum(s, 255)
    tl.store(d_ptr + offs, s.to(tl.uint8), mask=mask)


# ---------------------------------------------------------------------------
# Generic paths: other dtypes (mode 0 float, mode 2 integral) and any
# non-contiguous / broadcast operand layout.
# ---------------------------------------------------------------------------
@triton.jit
def _combine(
    x, y, alpha, MODE: tl.constexpr, HAS_ALPHA: tl.constexpr, CAST_F32: tl.constexpr
):
    if MODE == 0:
        if CAST_F32:
            x = x.to(tl.float32)
            y = y.to(tl.float32)
        if HAS_ALPHA:
            y = y * alpha
        out = x + y
    elif MODE == 1:
        xi = x.to(tl.int32)
        yi = y.to(tl.int32)
        if HAS_ALPHA:
            yi = yi * alpha.to(tl.int32)
        s = xi + yi
        s = tl.minimum(s, 255)
        out = s.to(tl.uint8)
    else:
        if HAS_ALPHA:
            y = y * alpha
        out = x + y
    return out


@triton.jit
def _add_generic_kernel(
    a_ptr,
    b_ptr,
    d_ptr,
    numel,
    alpha,
    MODE: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    CAST_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    if MODE == 0:
        x = tl.load(a_ptr + offs, mask=mask, other=0.0)
        y = tl.load(b_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(a_ptr + offs, mask=mask, other=0)
        y = tl.load(b_ptr + offs, mask=mask, other=0)
    out = _combine(x, y, alpha, MODE, HAS_ALPHA, CAST_F32)
    tl.store(d_ptr + offs, out, mask=mask)


# Generic strided / broadcast path: dimensions are right-padded to 4 with
# size 1 and stride 0, so a single straight-line decomposition covers rank<=4.
@triton.jit
def _add_strided4_kernel(
    a_ptr,
    b_ptr,
    d_ptr,
    numel,
    alpha,
    s0,
    s1,
    s2,
    s3,
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    b2,
    b3,
    d0,
    d1,
    d2,
    d3,
    MODE: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    CAST_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    rem = offs
    i3 = rem % s3
    rem = rem // s3
    i2 = rem % s2
    rem = rem // s2
    i1 = rem % s1
    i0 = rem // s1
    a_off = i0 * a0 + i1 * a1 + i2 * a2 + i3 * a3
    b_off = i0 * b0 + i1 * b1 + i2 * b2 + i3 * b3
    d_off = i0 * d0 + i1 * d1 + i2 * d2 + i3 * d3
    if MODE == 0:
        x = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        y = tl.load(b_ptr + b_off, mask=mask, other=0.0)
    else:
        x = tl.load(a_ptr + a_off, mask=mask, other=0)
        y = tl.load(b_ptr + b_off, mask=mask, other=0)
    out = _combine(x, y, alpha, MODE, HAS_ALPHA, CAST_F32)
    tl.store(d_ptr + d_off, out, mask=mask)


def _pad4(shape, strides):
    pad = 4 - len(shape)
    shape = [1] * pad + list(shape)
    strides = [0] * pad + list(strides)
    return shape, strides


def _bcast_meta(shape_a, shape_b):
    """Right-aligned broadcast shape of two shapes."""
    ra, rb = len(shape_a), len(shape_b)
    n = max(ra, rb)
    out = []
    for k in range(n):
        da = shape_a[ra - n + k] if ra - n + k >= 0 else 1
        db = shape_b[rb - n + k] if rb - n + k >= 0 else 1
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        else:
            raise ValueError("incompatible shapes for add")
    return out


def _rel_strides(t, out_shape):
    """Strides of t laid out against out_shape (0 on broadcast dims)."""
    n = len(out_shape)
    rt = t.dim()
    st = [0] * n
    for k in range(n):
        tk = rt - n + k
        if tk >= 0 and t.shape[tk] != 1:
            st[k] = t.stride(tk)
    return st


def add(a, b, dst=None, alpha=1, **kwargs):
    with torch.cuda.device(a.device):
        return _run(a, b, dst, alpha, **kwargs)


def _run(a, b, dst=None, alpha=1, **kwargs):
    if dst is None:
        dst = kwargs.pop("out", None)
    if "alpha" in kwargs:
        alpha = kwargs.pop("alpha")
    if kwargs:
        raise TypeError("unexpected keyword arguments: %s" % sorted(kwargs))
    # tolerate the torch.add-style positional scalar form: run(a, b, alpha)
    if dst is not None and not torch.is_tensor(dst):
        alpha = dst
        dst = None

    if dst is None or dst.shape != a.shape:
        dst = torch.empty_like(a)

    numel = a.numel()
    if numel == 0:
        return dst

    dt = a.dtype
    if dt == torch.uint8:
        mode = 1
        cast_f32 = False
        has_alpha = int(alpha) != 1
        alpha_v = int(alpha)
    elif dt.is_floating_point:
        mode = 0
        cast_f32 = dt.itemsize < 4
        has_alpha = float(alpha) != 1.0
        alpha_v = float(alpha)
    else:
        mode = 2
        cast_f32 = False
        has_alpha = int(alpha) != 1
        alpha_v = int(alpha)

    same_shape = a.shape == b.shape
    contiguous = a.is_contiguous() and b.is_contiguous() and dst.is_contiguous()

    if same_shape and contiguous:
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK"]),)
        if dt == torch.float32:
            _add_f32_kernel[grid](a, b, dst, numel, alpha_v, HAS_ALPHA=has_alpha)
            return dst
        if dt == torch.uint8:
            _add_u8_kernel[grid](a, b, dst, numel, alpha_v, HAS_ALPHA=has_alpha)
            return dst
        block = 1024
        _add_generic_kernel[(triton.cdiv(numel, block),)](
            a,
            b,
            dst,
            numel,
            alpha_v,
            MODE=mode,
            HAS_ALPHA=has_alpha,
            CAST_F32=cast_f32,
            BLOCK=block,
            num_warps=4,
        )
        return dst

    # generic strided / broadcast path
    out_shape = _bcast_meta(tuple(a.shape), tuple(b.shape))
    if len(out_shape) > 4:
        raise NotImplementedError(
            "add: rank > 4 with non-contiguous or broadcast operands"
        )
    out_numel = 1
    for s in out_shape:
        out_numel *= s
    if tuple(dst.shape) != tuple(out_shape):
        dst = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    os_, _ = _pad4(out_shape, [0] * len(out_shape))
    a_s = _rel_strides(a, out_shape)
    b_s = _rel_strides(b, out_shape)
    d_s = _rel_strides(dst, out_shape)
    _, a_s = _pad4(out_shape, a_s)
    _, b_s = _pad4(out_shape, b_s)
    _, d_s = _pad4(out_shape, d_s)

    block = 1024
    _add_strided4_kernel[(triton.cdiv(out_numel, block),)](
        a,
        b,
        dst,
        out_numel,
        alpha_v,
        os_[3],
        os_[2],
        os_[1],
        os_[0],
        a_s[3],
        a_s[2],
        a_s[1],
        a_s[0],
        b_s[3],
        b_s[2],
        b_s[1],
        b_s[0],
        d_s[3],
        d_s[2],
        d_s[1],
        d_s[0],
        MODE=mode,
        HAS_ALPHA=has_alpha,
        CAST_F32=cast_f32,
        BLOCK=block,
        num_warps=4,
    )
    return dst
