"""Shared case metadata and inputs for single-channel FlagCV profiles."""

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Case:
    shape: tuple[int, int]
    dtype: str
    boundary: bool = False
    extra: tuple = ()
    parameters: tuple = ()
    variant: str = ""

    @property
    def id(self):
        height, width = self.shape
        return f"{height}x{width}::{self.dtype}{self.variant}"


@dataclass(frozen=True)
class Profile:
    name: str
    tensors: tuple[str, ...]
    timing: Sequence[Case]
    correctness: Sequence[Case]
    boundary_inputs: tuple
    parameters: dict = field(default_factory=dict)

    def describe(self, case, nodeid, ordinal):
        return {
            "case_id": f"{nodeid}::core::{case.dtype}::{ordinal}",
            "ordinal": ordinal,
            "dtype": f"torch.{case.dtype}",
            "shape": {name: list(case.shape) for name in self.tensors},
            "params": {
                "channels": 1,
                "mask": None,
                "contiguous": True,
                **self.parameters,
                **dict(case.parameters),
            },
        }

    def make_inputs(self, case, device):
        import torch

        dtype = getattr(torch, case.dtype)
        if case.boundary:
            sources = tuple(
                torch.tensor(values, dtype=dtype, device=device)
                for values in self.boundary_inputs
            )
        else:
            generator = torch.Generator(device="cpu").manual_seed(0)
            if dtype == torch.uint8:
                sources = tuple(
                    torch.randint(
                        0, 256, case.shape, dtype=dtype, generator=generator
                    ).to(device)
                    for _ in self.tensors[:-1]
                )
            else:
                sources = tuple(
                    torch.randn(case.shape, dtype=dtype, generator=generator).to(device)
                    for _ in self.tensors[:-1]
                )
        dst = torch.empty(case.shape, dtype=dtype, device=device)
        return (*sources, dst, *case.extra)


SHAPES = (
    (64, 64),
    (61, 67),
    (224, 224),
    (257, 509),
    (480, 640),
    (720, 1280),
    (1080, 1920),
)
DEGENERATE_SHAPES = ((1, 1), (1, 257), (257, 1))


def image_cases(shapes, dtypes):
    return tuple(Case(shape, dtype) for shape in shapes for dtype in dtypes)
