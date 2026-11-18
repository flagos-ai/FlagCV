"""Small tensor generators for operator tests and benchmarks."""


def generate_tensor_input(shape, dtype, device, generator):
    """Sample an input on the CPU before moving it to the requested device."""
    import torch

    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype, generator=generator).to(device)
    return torch.randn(shape, dtype=dtype, generator=generator).to(device)
