"""List, probe and time the bitwise_not workloads."""

import pytest

from helpers.benchmark import run
from workloads.bitwise_not import OP_NAME, TIMING_CASES, describe, make_inputs


@pytest.mark.bitwise_not
def test_bitwise_not_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        cases=TIMING_CASES,
        make_inputs=make_inputs,
        describe=describe,
    )
