"""Verify the narrow OpenCV subtract contract."""

import pytest

from helpers.flagcv_correctness import check
from workloads.elementwise import correctness_cases


@pytest.mark.subtract
@pytest.mark.parametrize(
    "case", correctness_cases("subtract"), ids=lambda case: case.id
)
def test_subtract(case, candidate, native_reference, pytestconfig):
    check("subtract", case, candidate, native_reference, pytestconfig)
