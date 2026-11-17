"""List, probe and time the flip workloads."""

import pytest

from helpers.flagcv_benchmark import run


@pytest.mark.flip
def test_flip_benchmark(request, pytestconfig):
    run("flip", request, pytestconfig)
