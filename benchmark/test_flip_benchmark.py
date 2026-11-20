"""List, probe and time the flip workloads."""

import pytest

from helpers.benchmark import run
from workloads.flip import OP_NAME, TIMING_CASES, describe, make_inputs


@pytest.mark.flip
def test_flip_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        cases=TIMING_CASES,
        make_inputs=make_inputs,
        describe=describe,
        baseline_tensor="src",
    )
