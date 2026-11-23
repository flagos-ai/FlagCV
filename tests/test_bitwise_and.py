"""Verify the single-channel OpenCV-compatible bitwise_and profile."""

from functools import partial

import pytest

from helpers.correctness import check
from helpers.tolerance import assert_exact
from workloads.bitwise_and import CORRECTNESS_CASES, OP_NAME, case_id, make_inputs


def _oracle(src1, src2):
    import numpy as np

    return np.bitwise_and(src1, src2)


@pytest.mark.bitwise_and
@pytest.mark.parametrize(
    ("shape", "dtype", "fixed_inputs"),
    [pytest.param(*case, id=case_id(case)) for case in CORRECTNESS_CASES],
)
def test_bitwise_and(
    shape, dtype, fixed_inputs, candidate, native_reference, pytestconfig
):
    case = (shape, dtype, fixed_inputs)
    check(
        OP_NAME,
        partial(make_inputs, case),
        candidate,
        native_reference,
        pytestconfig,
        _oracle,
        source_count=2,
        assert_output=assert_exact,
    )
