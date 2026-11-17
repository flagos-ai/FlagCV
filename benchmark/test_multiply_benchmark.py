"""List, probe and time the multiply workloads."""

import pytest

from helpers.flagcv_benchmark import run


@pytest.mark.multiply
def test_multiply_benchmark(request, pytestconfig):
    run("multiply", request, pytestconfig)
