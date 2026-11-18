"""Correctness and timing workloads for flip."""

from helpers.inputs import generate_tensor_input

OP_NAME = "flip"
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

_BOUNDARY_INPUT = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]

TIMING_CASES = tuple(
    (shape, dtype, code, None)
    for shape in _TIMING_SHAPES
    for dtype in ("float32", "uint8")
    for code in (0, 1, -1)
)
CORRECTNESS_CASES = (
    *TIMING_CASES,
    *(((3, 5), "uint8", code, _BOUNDARY_INPUT) for code in (0, 2, -2)),
    *((shape, "uint8", code, None) for shape in _SMALL_SHAPES for code in (0, 1, -1)),
)


def case_id(case):
    (height, width), dtype, code, _ = case
    return f"{height}x{width}::{dtype}::flip{code}"


def describe(case):
    shape, dtype, code, _ = case
    shapes = dict.fromkeys(("src", "dst"), shape)
    params = {"channels": 1, "mask": None, "contiguous": True, "flip_code": code}
    return dtype, shapes, params


def make_inputs(case, device):
    import torch

    shape, dtype, code, fixed_input = case
    dtype = getattr(torch, dtype)
    if fixed_input is not None:
        src = torch.tensor(fixed_input, dtype=dtype, device=device)
    else:
        generator = torch.Generator(device="cpu").manual_seed(0)
        src = generate_tensor_input(shape, dtype, device, generator)
    return src, torch.empty_like(src), code
