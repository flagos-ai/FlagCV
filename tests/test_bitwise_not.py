"""Verify the single-channel OpenCV-compatible bitwise_not profile."""

from functools import partial

import pytest

from helpers.correctness import check
from helpers.tolerance import assert_exact
from workloads.bitwise_not import CORRECTNESS_CASES, OP_NAME, case_id, make_inputs


def _oracle(src):
    import numpy as np

    return np.bitwise_not(src)


@pytest.mark.bitwise_not
@pytest.mark.parametrize(
    ("shape", "dtype", "fixed_input"),
    [pytest.param(*case, id=case_id(case)) for case in CORRECTNESS_CASES],
)
def test_bitwise_not(
    shape, dtype, fixed_input, candidate, native_reference, pytestconfig
):
    case = (shape, dtype, fixed_input)
    check(
        OP_NAME,
        partial(make_inputs, case),
        candidate,
        native_reference,
        pytestconfig,
        _oracle,
        source_count=1,
        assert_output=assert_exact,
    )
