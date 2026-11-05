"""Shared workloads and input builder for the first OpenCV CUDA add profile."""

from dataclasses import dataclass

OP_NAME = "add"
SHAPES = ((64, 64), (480, 640), (1080, 1920), (2160, 3840))
DTYPES = ("float32", "uint8")


@dataclass(frozen=True)
class Case:
    shape: tuple
    dtype: str

    @property
    def id(self):
        height, width = self.shape
        return f"add::{height}x{width}::{self.dtype}"

    def describe(self, nodeid, ordinal):
        return {
            "case_id": f"{nodeid}::core::{self.dtype}::{ordinal}",
            "ordinal": ordinal,
            "dtype": f"torch.{self.dtype}",
            "shape": {name: list(self.shape) for name in ("a", "b", "dst")},
            "params": {"channels": 1, "mask": None, "contiguous": True},
        }


CASES = tuple(Case(shape, dtype) for shape in SHAPES for dtype in DTYPES)


def make_inputs(case, device):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(0)
    if case.dtype == "uint8":
        inputs = [
            torch.randint(0, 256, case.shape, dtype=torch.uint8, generator=generator)
            for _ in range(2)
        ]
    else:
        inputs = [
            torch.randn(case.shape, dtype=torch.float32, generator=generator)
            for _ in range(2)
        ]
    a, b = (tensor.to(device) for tensor in inputs)
    return a, b, torch.empty_like(a)
