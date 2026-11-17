"""Torch reference for OpenCV multiplication with scale fixed to one."""


def reference(a, b, dst):
    import torch

    if a.dtype == torch.uint8:
        # Each integer product fits exactly in float32 before saturation.
        values = (a.float() * b.float()).clamp_(0, 255)
    else:
        values = a * b
    dst.copy_(values)
    return dst
