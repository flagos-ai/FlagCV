import torch
import triton
import triton.language as tl


@triton.jit
def _xor(a, b, IS_F32: tl.constexpr):
    if IS_F32:
        result = (a.to(tl.int32, bitcast=True) ^ b.to(tl.int32, bitcast=True)).to(
            tl.float32, bitcast=True
        )
    else:
        result = a ^ b
    return result


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": b}, num_warps=w)
        for b in (256, 512, 1024, 2048, 4096)
        for w in (2, 4, 8)
    ],
    key=["N", "HAS_MASK", "IS_F32"],
    warmup=1,
    rep=5,
)
@triton.jit
def _xor_kernel(
    p_src1,
    p_src2,
    p_dst,
    p_mask,
    N,
    HAS_MASK: tl.constexpr,
    IS_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    a = tl.load(p_src1 + off, mask=m, other=0)
    b = tl.load(p_src2 + off, mask=m, other=0)
    r = _xor(a, b, IS_F32)
    if HAS_MASK:
        mv = tl.load(p_mask + off, mask=m, other=0)
        tl.store(p_dst + off, r, mask=m & (mv != 0))
    else:
        tl.store(p_dst + off, r, mask=m)


@triton.jit
def _xor_strided_kernel(
    src1,
    src2,
    dst,
    mask,
    size,
    columns,
    channels,
    a_row,
    a_column,
    a_channel,
    b_row,
    b_column,
    b_channel,
    d_row,
    d_column,
    d_channel,
    m_row,
    m_column,
    HAS_MASK: tl.constexpr,
    IS_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < size
    pixel = offset // channels
    row = pixel // columns
    column = pixel % columns
    channel = offset % channels
    a = tl.load(
        src1 + row * a_row + column * a_column + channel * a_channel,
        mask=valid,
        other=0,
    )
    b = tl.load(
        src2 + row * b_row + column * b_column + channel * b_channel,
        mask=valid,
        other=0,
    )
    if HAS_MASK:
        valid = valid & (
            tl.load(mask + row * m_row + column * m_column, mask=valid, other=0) != 0
        )
    tl.store(
        dst + row * d_row + column * d_column + channel * d_channel,
        _xor(a, b, IS_F32),
        mask=valid,
    )


def bitwise_xor(src1, src2, dst, mask=None):
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


def _run(src1, src2, dst, mask=None):
    n = src1.numel()
    if n == 0:
        return dst
    if not all(t.is_contiguous() for t in (src1, src2, dst)) or (
        mask is not None and not mask.is_contiguous()
    ):
        channels = src1.size(2) if src1.ndim == 3 else 1
        strides = [
            t.stride() if t.ndim == 3 else (*t.stride(), 0) for t in (src1, src2, dst)
        ]
        block = 256
        _xor_strided_kernel[(triton.cdiv(n, block),)](
            src1,
            src2,
            dst,
            dst if mask is None else mask,
            n,
            src1.size(1),
            channels,
            *strides[0],
            *strides[1],
            *strides[2],
            *((0, 0) if mask is None else mask.stride()),
            HAS_MASK=mask is not None,
            IS_F32=src1.dtype == torch.float32,
            BLOCK=block,
        )
        return dst
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _xor_kernel[grid](
        src1,
        src2,
        dst,
        dst if mask is None else mask,
        n,
        HAS_MASK=mask is not None,
        IS_F32=(src1.dtype == torch.float32),
    )
    return dst
