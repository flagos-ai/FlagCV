"""List, probe and time the subtract workloads."""

import pytest

from helpers.flagcv_benchmark import run


@pytest.mark.subtract
def test_subtract_benchmark(request, pytestconfig):
    run("subtract", request, pytestconfig)
