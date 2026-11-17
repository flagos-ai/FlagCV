"""Verify the narrow OpenCV bitwise_and contract."""

import pytest

from helpers.flagcv_correctness import check
from workloads.elementwise import correctness_cases


@pytest.mark.bitwise_and
@pytest.mark.parametrize(
    "case", correctness_cases("bitwise_and"), ids=lambda case: case.id
)
def test_bitwise_and(case, candidate, native_reference, pytestconfig):
    check("bitwise_and", case, candidate, native_reference, pytestconfig)
