"""Correctness and timing workloads for multiply."""

from helpers.inputs import generate_tensor_input

OP_NAME = "multiply"
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

TIMING_CASES = tuple(
    (shape, dtype, None) for shape in _TIMING_SHAPES for dtype in ("float32", "uint8")
)
CORRECTNESS_CASES = (
    *TIMING_CASES,
    ((1, 5), "uint8", ([[0, 1, 5, 200, 255]], [[255, 255, 3, 2, 255]])),
    *((shape, "uint8", None) for shape in _SMALL_SHAPES),
)


def case_id(case):
    (height, width), dtype, _ = case
    return f"{height}x{width}::{dtype}"


def describe(case):
    shape, dtype, _ = case
    shapes = dict.fromkeys(("a", "b", "dst"), shape)
    params = {"channels": 1, "mask": None, "contiguous": True, "scale": 1.0}
    return dtype, shapes, params


def make_inputs(case, device):
    import torch

    shape, dtype, fixed_inputs = case
    dtype = getattr(torch, dtype)
    if fixed_inputs is not None:
        a, b = (
            torch.tensor(values, dtype=dtype, device=device) for values in fixed_inputs
        )
    else:
        generator = torch.Generator(device="cpu").manual_seed(0)
        a, b = (
            generate_tensor_input(shape, dtype, device, generator) for _ in range(2)
        )
    return a, b, torch.empty_like(a)
