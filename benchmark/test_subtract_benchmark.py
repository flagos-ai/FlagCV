"""List, probe and time the subtract workloads."""

import pytest

from helpers.benchmark import run
from workloads.subtract import OP_NAME, TIMING_CASES, describe, make_inputs


@pytest.mark.subtract
def test_subtract_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        cases=TIMING_CASES,
        make_inputs=make_inputs,
        describe=describe,
    )
