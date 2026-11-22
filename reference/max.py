"""Torch reference for OpenCV CUDA extrema."""

import torch


def reference(src1, src2, dst):
    if src1.dtype == torch.uint16:
        # Use signed storage for copies, but compare the full unsigned values.
        left = src1.view(torch.int16).int() & 0xFFFF
        right = src2.view(torch.int16).int() & 0xFFFF
        dst.view(torch.int16).copy_(torch.fmax(left, right).to(torch.int16))
        return dst
    result = torch.fmax(src1, src2)
    if src1.dtype.is_floating_point:
        # CUDA min picks negative zero if either zero is negative; CUDA max
        # does so only when both zeros are negative.
        a_negative, b_negative = torch.signbit(src1), torch.signbit(src2)
        negative_zero = a_negative & b_negative
        zero = torch.where(
            negative_zero, result.new_tensor(-0.0), result.new_tensor(0.0)
        )
        result = torch.where((src1 == 0) & (src2 == 0), zero, result)
    dst.copy_(result)
    return dst
