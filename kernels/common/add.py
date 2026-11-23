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
def _add_f32_kernel(a_ptr, b_ptr, d_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    y = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(d_ptr + offs, x + y, mask=mask)


@triton.autotune(configs=U8_CONFIGS, key=["numel"])
@triton.jit
def _add_u8_kernel(a_ptr, b_ptr, d_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int32)
    y = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int32)
    s = x + y
    s = tl.minimum(s, 255)
    tl.store(d_ptr + offs, s.to(tl.uint8), mask=mask)


def add(a, b, dst):
    """Add equally shaped contiguous single-channel images into dst."""
    if a.ndim != 2 or a.dtype not in (torch.uint8, torch.float32):
        raise ValueError("add requires a 2-D uint8 or float32 image")
    if any(
        (t.shape, t.dtype, t.device) != (a.shape, a.dtype, a.device) for t in (b, dst)
    ):
        raise ValueError(
            "add inputs and destination must match in shape, dtype, and device"
        )
    if not all(t.is_contiguous() for t in (a, b, dst)):
        raise ValueError("add requires contiguous images")
    size = a.numel()
    if size:
        kernel = _add_u8_kernel if a.dtype == torch.uint8 else _add_f32_kernel
        with torch.cuda.device(a.device):
            kernel[lambda meta: (triton.cdiv(size, meta["BLOCK"]),)](a, b, dst, size)
    return dst
