"""List, probe and time the multiply workloads."""

import pytest

from helpers.benchmark import run
from workloads.multiply import OP_NAME, TIMING_CASES, describe, make_inputs


@pytest.mark.multiply
def test_multiply_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        cases=TIMING_CASES,
        make_inputs=make_inputs,
        describe=describe,
        baseline_tensor="a",
    )
