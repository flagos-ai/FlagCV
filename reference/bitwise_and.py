"""Torch reference for unmasked OpenCV uint8 AND."""


def reference(src1, src2, dst):
    import torch

    dst.copy_(torch.bitwise_and(src1, src2))
    return dst
