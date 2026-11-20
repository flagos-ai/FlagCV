"""Correctness and timing workloads for add."""

from helpers.inputs import generate_tensor_input

OP_NAME = "add"
_TIMING_SHAPES = ((64, 64), (480, 640), (1080, 1920), (2160, 3840))

TIMING_CASES = tuple(
    (shape, dtype, None) for shape in _TIMING_SHAPES for dtype in ("float32", "uint8")
)
SATURATION_CASE = ((1, 3), "uint8", ([[250, 1, 255]], [[10, 2, 255]]))
CORRECTNESS_CASES = (*TIMING_CASES, SATURATION_CASE)


def case_id(case):
    (height, width), dtype, _ = case
    return f"add::{height}x{width}::{dtype}"


def describe(case):
    shape, dtype, _ = case
    shapes = dict.fromkeys(("a", "b", "dst"), shape)
    params = {"channels": 1, "mask": None, "contiguous": True}
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
            generate_tensor_input(shape, dtype, device, generator)
            for _ in range(2)
        )
    return a, b, torch.empty_like(a)
