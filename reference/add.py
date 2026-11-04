"""Default Torch reference for OpenCV-compatible add."""

def reference(a, b, dst):
    import torch

    if a.dtype == torch.uint8:
        dst.copy_((a.to(torch.int16) + b.to(torch.int16)).clamp_(0, 255))
    else:
        torch.add(a, b, out=dst)
    return dst
