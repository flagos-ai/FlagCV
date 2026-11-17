"""List, probe and time the bitwise_and workloads."""

import pytest

from helpers.flagcv_benchmark import run


@pytest.mark.bitwise_and
def test_bitwise_and_benchmark(request, pytestconfig):
    run("bitwise_and", request, pytestconfig)
