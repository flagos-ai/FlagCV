"""Correctness and timing workloads for bitwise_not."""

from helpers.inputs import generate_tensor_input

OP_NAME = "bitwise_not"
_TIMING_SHAPES = (
    (64, 64),
    (61, 67),
    (224, 224),
    (257, 509),
    (480, 640),
    (720, 1280),
    (1080, 1920),
)
_SMALL_SHAPES = ((1, 1), (1, 257), (257, 1))

TIMING_CASES = tuple((shape, "uint8", None) for shape in _TIMING_SHAPES)
CORRECTNESS_CASES = (
    *TIMING_CASES,
    ((1, 5), "uint8", [[0, 1, 170, 85, 255]]),
    *((shape, "uint8", None) for shape in _SMALL_SHAPES),
)


def case_id(case):
    (height, width), dtype, _ = case
    return f"{height}x{width}::{dtype}"


def describe(case):
    shape, dtype, _ = case
    shapes = dict.fromkeys(("src", "dst"), shape)
    params = {"channels": 1, "mask": None, "contiguous": True}
    return dtype, shapes, params


def make_inputs(case, device):
    import torch

    shape, dtype, fixed_input = case
    dtype = getattr(torch, dtype)
    if fixed_input is not None:
        src = torch.tensor(fixed_input, dtype=dtype, device=device)
    else:
        generator = torch.Generator(device="cpu").manual_seed(0)
        src = generate_tensor_input(shape, dtype, device, generator)
    return src, torch.empty_like(src)
