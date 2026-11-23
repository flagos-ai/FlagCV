"""Torch reference for unmasked OpenCV uint8 complement."""


def reference(src, dst):
    import torch

    dst.copy_(torch.bitwise_not(src))
    return dst
