"""List, probe and time the bitwise_not workloads."""

import pytest

from helpers.flagcv_benchmark import run


@pytest.mark.bitwise_not
def test_bitwise_not_benchmark(request, pytestconfig):
    run("bitwise_not", request, pytestconfig)
