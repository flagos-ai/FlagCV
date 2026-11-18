"""Verify the single-channel OpenCV-compatible subtract profile."""

from functools import partial

import pytest

from helpers.correctness import check
from helpers.tolerance import assert_one_lsb, assert_relative_close
from workloads.subtract import CORRECTNESS_CASES, OP_NAME, case_id, make_inputs


def _oracle(a, b):
    import numpy as np

    if a.dtype == np.uint8:
        values = np.subtract(a.astype(np.int32), b.astype(np.int32))
        return values.clip(0, 255).astype(np.uint8)
    return np.subtract(a, b)


@pytest.mark.subtract
@pytest.mark.parametrize(
    ("shape", "dtype", "fixed_inputs"),
    [pytest.param(*case, id=case_id(case)) for case in CORRECTNESS_CASES],
)
def test_subtract(
    shape, dtype, fixed_inputs, candidate, native_reference, pytestconfig
):
    case = (shape, dtype, fixed_inputs)
    assert_output = assert_one_lsb if dtype == "uint8" else assert_relative_close
    check(
        OP_NAME,
        partial(make_inputs, case),
        candidate,
        native_reference,
        pytestconfig,
        _oracle,
        source_count=2,
        assert_output=assert_output,
    )
