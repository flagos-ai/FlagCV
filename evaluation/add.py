"""Shared workload and reference for the first OpenCV CUDA add profile."""

import math
from dataclasses import dataclass

OP_NAME = "add"
SHAPES = ((64, 64), (480, 640), (1080, 1920), (2160, 3840))
DTYPES = ("float32", "uint8")
CONTRACT = {
    "native_api": "cv::cuda::add",
    "channels": 1,
    "mask": None,
    "output_preallocated": True,
}


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


def lookup_baseline(data, case):
    """Match the narrow ops/opencv/add.py profile and return microseconds."""
    dtype = f"torch.{case.dtype}"
    record = (
        data.get(OP_NAME, {})
        .get("shapes", {})
        .get(str(list(case.shape)), {})
        .get(dtype)
    )
    if record is None:
        return None, "missing add/shape/dtype record"
    config = record.get("config", {})
    required = {**CONTRACT, "output_dtype": dtype}
    if any(
        key not in config or config[key] != value for key, value in required.items()
    ):
        return None, "incompatible OpenCV CUDA add configuration"
    latency = record.get("T_us")
    if type(latency) not in (float, int) or not math.isfinite(latency) or latency <= 0:
        return None, "missing or invalid T_us"
    return latency, None
