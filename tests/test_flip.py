"""Verify the narrow OpenCV flip contract."""

import pytest

from helpers.flagcv_correctness import check
from workloads.elementwise import correctness_cases


@pytest.mark.flip
@pytest.mark.parametrize("case", correctness_cases("flip"), ids=lambda case: case.id)
def test_flip(case, candidate, native_reference, pytestconfig):
    check("flip", case, candidate, native_reference, pytestconfig)
