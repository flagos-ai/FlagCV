"""Verify the narrow OpenCV bitwise_not contract."""

import pytest

from helpers.flagcv_correctness import check
from workloads.elementwise import correctness_cases


@pytest.mark.bitwise_not
@pytest.mark.parametrize(
    "case", correctness_cases("bitwise_not"), ids=lambda case: case.id
)
def test_bitwise_not(case, candidate, native_reference, pytestconfig):
    check("bitwise_not", case, candidate, native_reference, pytestconfig)
