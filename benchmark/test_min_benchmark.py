"""List and time the min workloads."""

import pytest
from helpers.benchmark import run
from workloads.min import OP_NAME, TIMING_CASES, describe, make_inputs


@pytest.mark.min
def test_min_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        cases=TIMING_CASES,
        make_inputs=make_inputs,
        describe=describe,
        baseline_tensor="src1",
    )
