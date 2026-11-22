"""List and time the bitwise_or workloads."""

import pytest
from helpers.benchmark import run
from workloads.bitwise_or import OP_NAME, TIMING_CASES, describe, make_inputs


@pytest.mark.bitwise_or
def test_bitwise_or_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        cases=TIMING_CASES,
        make_inputs=make_inputs,
        describe=describe,
        baseline_tensor="src1",
    )
