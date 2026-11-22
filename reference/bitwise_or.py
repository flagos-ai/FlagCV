"""Torch reference for OpenCV CUDA bitwise operations."""

import torch


def reference(src1, src2, dst, mask=None):
    # CV_32F bitwise methods use the encoding, not floating point arithmetic.
    storage_type = {torch.float32: torch.int32, torch.uint16: torch.int16}.get(
        src1.dtype, src1.dtype
    )
    target = dst.view(storage_type)
    result = torch.bitwise_or(src1.view(storage_type), src2.view(storage_type))
    if mask is not None:
        active = mask != 0
        if src1.ndim == 3:
            active = active.unsqueeze(-1)
        result = torch.where(active, result, target)
    target.copy_(result)
    return dst
