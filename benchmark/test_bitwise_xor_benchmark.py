"""List and time the bitwise_xor workloads."""

import pytest
from helpers.benchmark import run
from workloads.bitwise_xor import OP_NAME, TIMING_CASES, describe, make_inputs


@pytest.mark.bitwise_xor
def test_bitwise_xor_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        cases=TIMING_CASES,
        make_inputs=make_inputs,
        describe=describe,
        baseline_tensor="src1",
    )
