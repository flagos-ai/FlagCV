"""Torch reference for saturating OpenCV subtraction."""


def reference(a, b, dst):
    import torch

    if a.dtype == torch.uint8:
        values = (a.to(torch.int16) - b.to(torch.int16)).clamp_(0, 255)
    else:
        values = a - b
    dst.copy_(values)
    return dst
