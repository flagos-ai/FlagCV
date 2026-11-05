"""List, probe or measure the shared OpenCV-compatible add workloads."""

import json
import math
from contextlib import nullcontext

import pytest

from conftest import DummyReached, dummy, emit
from workloads.add import CASES, OP_NAME, make_inputs

CONTRACT = {
    "native_api": "cv::cuda::add",
    "channels": 1,
    "mask": None,
    "output_preallocated": True,
}


def _lookup_baseline(data, case):
    """Match the narrow ops/opencv/add.py profile and return microseconds."""
    dtype = f"torch.{case.dtype}"
    record = (
        data.get(OP_NAME, {})
        .get("shapes", {})
        .get(str(list(case.shape)), {})
        .get(dtype)
    )
    if record is None:
        return None, "missing add/shape/dtype record"
    config = record.get("config", {})
    required = {**CONTRACT, "output_dtype": dtype}
    if any(
        key not in config or config[key] != value for key, value in required.items()
    ):
        return None, "incompatible OpenCV CUDA add configuration"
    latency = record.get("T_us")
    if type(latency) not in (float, int) or not math.isfinite(latency) or latency <= 0:
        return None, "missing or invalid T_us"
    return latency, None


def measure_us(function, inputs):
    import torch

    device = inputs[0].device
    if device.type == "cpu":
        pytest.fail("Device timing requires an accelerator")

    from triton.testing import do_bench

    with torch.get_device_module(device).device(device):
        return do_bench(lambda: function(*inputs), warmup=25, rep=100) * 1000


def _profile_case(config, case_id, candidate, warmup_inputs, capture_inputs):
    import torch

    device = capture_inputs[0].device
    backend = config._flagcv_runtime.vendor
    device_module = torch.get_device_module(device) if device.type != "cpu" else None
    with device_module.device(device) if device_module else nullcontext():
        for _ in range(config.getoption("profile_warmup")):
            candidate(*warmup_inputs)
        if device_module:
            device_module.synchronize()
        scope = config.hook.pytest_flaggems_profile_scope(
            backend=backend, case_id=case_id
        )
        with scope if scope is not None else nullcontext():
            for _ in range(config.getoption("profile_iterations")):
                candidate(*capture_inputs)
            if device_module:
                device_module.synchronize()


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
    device = pytestconfig.getoption("device")
    if pytestconfig.getoption("profile_only"):
        case = next(
            case for case, row in zip(CASES, cases) if row["case_id"] == selected[0]
        )
        _profile_case(
            pytestconfig,
            selected[0],
            candidate,
            make_inputs(case, device),
            make_inputs(case, device),
        )
        return

    baseline_path = pytestconfig.getoption("base_data")
    baseline = json.loads(baseline_path.read_text()) if baseline_path else None
    results = []
    for case, result in zip(CASES, cases):
        if selected and result["case_id"] not in selected:
            continue
        inputs = make_inputs(case, device)
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
                base_us, reason = _lookup_baseline(baseline, case)
                result.update(
                    baseline_us=base_us,
                    baseline_miss=reason,
                    raw_speedup=base_us / latency if base_us is not None else None,
                )
        results.append(result)
    emit(pytestconfig, {"op_name": OP_NAME, "cases": results})
