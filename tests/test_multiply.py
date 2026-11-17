"""Verify the narrow OpenCV multiply contract."""

import pytest

from helpers.flagcv_correctness import check
from workloads.elementwise import correctness_cases


@pytest.mark.multiply
@pytest.mark.parametrize(
    "case", correctness_cases("multiply"), ids=lambda case: case.id
)
def test_multiply(case, candidate, native_reference, pytestconfig):
    check("multiply", case, candidate, native_reference, pytestconfig)
