"""Workloads for the single-channel, independent-destination FlagCV profiles."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Case:
    shape: tuple[int, int]
    dtype: str
    flip_code: Optional[int] = None
    boundary: bool = False

    @property
    def id(self):
        height, width = self.shape
        suffix = f"::flip{self.flip_code}" if self.flip_code is not None else ""
        return f"{height}x{width}::{self.dtype}{suffix}"

    def describe(self, op_name, nodeid, ordinal):
        names = {
            "subtract": ("a", "b", "dst"),
            "multiply": ("a", "b", "dst"),
            "bitwise_and": ("src1", "src2", "dst"),
            "bitwise_not": ("src", "dst"),
            "flip": ("src", "dst"),
        }[op_name]
        params = {"channels": 1, "mask": None, "contiguous": True}
        if op_name == "multiply":
            params["scale"] = 1.0
        if op_name == "flip":
            params["flip_code"] = self.flip_code
        return {
            "case_id": f"{nodeid}::core::{self.dtype}::{ordinal}",
            "ordinal": ordinal,
            "dtype": f"torch.{self.dtype}",
            "shape": {name: list(self.shape) for name in names},
            "params": params,
        }


SHAPES = (
    (64, 64),
    (61, 67),
    (224, 224),
    (257, 509),
    (480, 640),
    (720, 1280),
    (1080, 1920),
)
_DEGENERATE_SHAPES = ((1, 1), (1, 257), (257, 1))
ARITHMETIC = tuple(
    Case(shape, dtype) for shape in SHAPES for dtype in ("float32", "uint8")
)
BITWISE = tuple(Case(shape, "uint8") for shape in SHAPES)
CASES = {
    "subtract": ARITHMETIC,
    "multiply": ARITHMETIC,
    "bitwise_and": BITWISE,
    "bitwise_not": BITWISE,
    "flip": tuple(
        Case(shape, dtype, flip_code)
        for shape in SHAPES
        for dtype in ("float32", "uint8")
        for flip_code in (0, 1, -1)
    ),
}

_BOUNDARY_INPUTS = {
    "subtract": ([[0, 1, 5, 255, 127]], [[1, 0, 5, 1, 128]]),
    "multiply": ([[0, 1, 5, 200, 255]], [[255, 255, 3, 2, 255]]),
    "bitwise_and": ([[255, 240, 165, 170, 0]], [[15, 255, 90, 15, 255]]),
    "bitwise_not": ([[0, 1, 170, 85, 255]],),
    "flip": ([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],),
}


def correctness_cases(op_name):
    if op_name == "flip":
        boundary = tuple(
            Case((3, 5), "uint8", code, boundary=True) for code in (0, 2, -2)
        )
        degenerate = tuple(
            Case(shape, "uint8", code)
            for shape in _DEGENERATE_SHAPES
            for code in (0, 1, -1)
        )
    else:
        boundary = (Case((1, 5), "uint8", boundary=True),)
        degenerate = tuple(Case(shape, "uint8") for shape in _DEGENERATE_SHAPES)
    return (*CASES[op_name], *boundary, *degenerate)


def make_inputs(op_name, case, device):
    import torch

    dtype = getattr(torch, case.dtype)
    source_count = 1 if op_name in ("bitwise_not", "flip") else 2
    if case.boundary:
        sources = tuple(
            torch.tensor(values, dtype=dtype, device=device)
            for values in _BOUNDARY_INPUTS[op_name]
        )
    else:
        generator = torch.Generator(device="cpu").manual_seed(0)
        if dtype == torch.uint8:
            sources = tuple(
                torch.randint(0, 256, case.shape, dtype=dtype, generator=generator).to(
                    device
                )
                for _ in range(source_count)
            )
        else:
            sources = tuple(
                torch.randn(case.shape, dtype=dtype, generator=generator).to(device)
                for _ in range(source_count)
            )
    dst = torch.empty(case.shape, dtype=dtype, device=device)
    extra = (case.flip_code,) if op_name == "flip" else ()
    return (*sources, dst, *extra)
