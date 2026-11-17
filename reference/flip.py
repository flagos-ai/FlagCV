"""Torch reference for the OpenCV flip direction convention."""


def reference(src, dst, flip_code):
    import torch

    axes = (0,) if flip_code == 0 else (1,) if flip_code > 0 else (0, 1)
    dst.copy_(torch.flip(src, axes))
    return dst
