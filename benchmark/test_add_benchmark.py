"""List, probe or measure the shared OpenCV-compatible add workloads."""

import json
import math

import pytest

from conftest import DummyReached, dummy, emit
from evaluation.add import CASES, OP_NAME, lookup_baseline, make_inputs


def measure_us(function, inputs):
    import torch

    device = inputs[0].device
    if device.type == "cpu":
        pytest.fail("Device timing requires an accelerator")

    from triton.testing import do_bench

    with torch.get_device_module(device).device(device):
        return do_bench(lambda: function(*inputs), warmup=25, rep=100) * 1000


@pytest.mark.add
def test_add_benchmark(request, pytestconfig):
    cases = [case.describe(request.node.nodeid, i) for i, case in enumerate(CASES)]
    if pytestconfig.getoption("list_cases"):
        emit(
            pytestconfig,
            {
                "schema_version": "flaggems.benchmark-case-list/v2",
                "benchmarks": [
                    {
                        "op_name": OP_NAME,
                        "level": "core",
                        "phase": "timing",
                        "cases": cases,
                    }
                ],
            },
        )
        return

    selected = pytestconfig.getoption("case_id")
    unknown = set(selected) - {case["case_id"] for case in cases}
    if unknown:
        raise pytest.UsageError(f"Unknown --case-id: {', '.join(sorted(unknown))}")
    candidate = request.getfixturevalue("candidate")
    baseline_path = pytestconfig.getoption("base_data")
    baseline = json.loads(baseline_path.read_text()) if baseline_path else None
    results = []
    for case, result in zip(CASES, cases):
        if selected and result["case_id"] not in selected:
            continue
        inputs = make_inputs(case, pytestconfig.getoption("device"))
        if candidate is dummy:
            with pytest.raises(DummyReached):
                candidate(*inputs)
            result["status"] = "reached"
        else:
            latency = measure_us(candidate, inputs)
            assert math.isfinite(latency) and latency > 0
            result.update(
                latency_us=latency,
                candidate_source=(
                    "reference"
                    if pytestconfig.getoption("ref_as_candidate")
                    else "override"
                ),
            )
            if baseline is not None:
                base_us, reason = lookup_baseline(baseline, case)
                result.update(
                    baseline_us=base_us,
                    baseline_miss=reason,
                    raw_speedup=base_us / latency if base_us is not None else None,
                )
        results.append(result)
    emit(pytestconfig, {"op_name": OP_NAME, "cases": results})
