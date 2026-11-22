"""Shared shape, depth and mask grid for OpenCV CUDA binary operations."""

from dataclasses import dataclass

OP_NAME = "bitwise_or"
DEPTHS = ("uint8", "int8", "uint16", "int16", "int32", "float32")


@dataclass(frozen=True)
class Case:
    shape: tuple[int, ...]
    dtype: str
    masked: bool = False
    special: bool = False

    @property
    def id(self):
        shape = "x".join(map(str, self.shape))
        return f"{shape}-{self.dtype}-mask{int(self.masked)}-special{int(self.special)}"


def _timing_cases():
    return (
        *(Case((64, 67), dtype) for dtype in DEPTHS),
        Case((512, 513, 3), "float32"),
        Case((512, 513), "uint8", masked=True),
        *(Case((31, 127), dtype) for dtype in DEPTHS),
        *(Case((256, 257), dtype) for dtype in DEPTHS),
        Case((31, 127), "int16", masked=True),
        Case((256, 257), "float32", masked=True),
    )


TIMING_CASES = _timing_cases()
CORRECTNESS_CASES = (
    *TIMING_CASES,
    *(Case((3, 5), dtype, masked=True) for dtype in DEPTHS if dtype != "float32"),
    Case((3, 5), "float32", masked=True, special=True),
)


def describe(case):
    shapes = dict.fromkeys(("src1", "src2", "dst"), case.shape)
    if case.masked:
        shapes["mask"] = case.shape[:2]
    params = {
        "channels": case.shape[2] if len(case.shape) == 3 else 1,
        "mask": "uint8" if case.masked else None,
        "contiguous": True,
    }
    return case.dtype, shapes, params


def make_inputs(case, device):
    import torch

    dtype = getattr(torch, case.dtype)
    generator = torch.Generator().manual_seed(913)
    if dtype.is_floating_point:
        a, b, dst = (
            torch.randn(case.shape, generator=generator).to(dtype) for _ in range(3)
        )
        if case.special:
            a.flatten()[:8] = a.new_tensor(
                [float("nan"), 2, float("inf"), 0.0, -0.0, -1, 3, -5]
            )
            b.flatten()[:8] = b.new_tensor(
                [1, float("nan"), -float("inf"), -0.0, 0.0, -2, 3, 7]
            )
    else:
        info = torch.iinfo(dtype)
        a, b, dst = (
            torch.randint(info.min, info.max + 1, case.shape, generator=generator).to(
                dtype
            )
            for _ in range(3)
        )
    mask = (
        torch.randint(0, 3, case.shape[:2], generator=generator).to(device, torch.uint8)
        if case.masked
        else None
    )
    return a.to(device), b.to(device), dst.to(device), mask
