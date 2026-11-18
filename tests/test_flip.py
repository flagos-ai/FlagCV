"""Verify the single-channel OpenCV-compatible flip profile."""

from functools import partial

import pytest

from helpers.correctness import check
from helpers.tolerance import assert_bitwise_equal
from workloads.flip import CORRECTNESS_CASES, OP_NAME, case_id, make_inputs


def _oracle(src, flip_code):
    import numpy as np

    axes = (0,) if flip_code == 0 else (1,) if flip_code > 0 else (0, 1)
    # NumPy flip produces a view with negative strides.
    return np.flip(src, axis=axes).copy()


@pytest.mark.flip
@pytest.mark.parametrize(
    ("shape", "dtype", "flip_code", "fixed_input"),
    [pytest.param(*case, id=case_id(case)) for case in CORRECTNESS_CASES],
)
def test_flip(
    shape, dtype, flip_code, fixed_input, candidate, native_reference, pytestconfig
):
    case = (shape, dtype, flip_code, fixed_input)
    check(
        OP_NAME,
        partial(make_inputs, case),
        candidate,
        native_reference,
        pytestconfig,
        _oracle,
        source_count=1,
        assert_output=assert_bitwise_equal,
    )


@pytest.mark.flip
def test_flip_preserves_float_bits(candidate, native_reference, pytestconfig):
    import numpy as np
    import torch

    def make_signed_zero_inputs(device):
        src = torch.tensor([[-0.0, 1.0, 0.0, -1.0]], dtype=torch.float32, device=device)
        return src, torch.empty_like(src), np.int64(-2)

    check(
        OP_NAME,
        make_signed_zero_inputs,
        candidate,
        native_reference,
        pytestconfig,
        _oracle,
        source_count=1,
        assert_output=assert_bitwise_equal,
    )
