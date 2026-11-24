"""List, probe and time the bitwise_and workloads."""

import pytest

from helpers.benchmark import run
from workloads.bitwise_and import OP_NAME, TIMING_CASES, describe, make_inputs


@pytest.mark.bitwise_and
def test_bitwise_and_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        cases=TIMING_CASES,
        make_inputs=make_inputs,
        describe=describe,
    )
