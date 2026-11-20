"""Run, list and profile FlagCV benchmarks."""

import json
import math
from contextlib import nullcontext
from functools import partial

import pytest

from conftest import emit


def _measure_ms(candidate, args):
    import torch

    device = args[0].device
    if device.type == "cpu":
        pytest.fail("Device timing requires an accelerator")

    from triton.testing import do_bench

    with torch.get_device_module(device).device(device):
        return do_bench(lambda: candidate(*args), warmup=25, rep=100)


def _profile(config, case_id, candidate, build):
    import torch

    warmup_inputs = build()
    capture_inputs = build()
    device = capture_inputs[0].device
    device_module = torch.get_device_module(device) if device.type != "cpu" else None
    device_scope = device_module.device(device) if device_module else nullcontext()
    with device_scope:
        for _ in range(config.getoption("profile_warmup")):
            candidate(*warmup_inputs)
        if device_module:
            device_module.synchronize()
        scope = config.hook.pytest_flaggems_profile_scope(
            backend=config._flagcv_runtime.vendor,
            case_id=case_id,
        )
        before = config._evaluation.calls
        with scope if scope is not None else nullcontext():
            for _ in range(config.getoption("profile_iterations")):
                candidate(*capture_inputs)
            if device_module:
                device_module.synchronize()
        assert (
            config._evaluation.calls > before
        ), "Profile did not execute its candidate"


def _lookup_baseline(data, op_name, match_shape, row):
    record = (
        data.get(op_name, {})
        .get("shapes", {})
        .get(str(match_shape), {})
        .get(row["dtype"])
    )
    if record is None:
        return None, "missing operator/shape/dtype record"
    config = record.get("config", {})
    required = {
        "native_api": f"cv::cuda::{op_name}",
        "output_preallocated": True,
        "output_dtype": row["dtype"],
        # The legacy OpenCV CUDA baseline does not record tensor contiguity.
        **{key: value for key, value in row["params"].items() if key != "contiguous"},
    }
    if any(
        key not in config or config[key] != value for key, value in required.items()
    ):
        return None, "incompatible OpenCV CUDA configuration"
    latency = record.get("T_us")
    if type(latency) not in (float, int) or not math.isfinite(latency) or latency <= 0:
        return None, "missing or invalid T_us"
    return latency, None


def run(request, config, *, op_name, cases, make_inputs, describe, baseline_tensor):
    """Run operator-owned cases using their input builder and metadata."""
    cases = tuple(cases)
    rows = []
    for index, case in enumerate(cases):
        dtype, shapes, params = describe(case)
        rows.append(
            {
                "case_id": f"{request.node.nodeid}::core::{dtype}::{index}",
                "ordinal": index,
                "dtype": f"torch.{dtype}",
                "shape": {name: list(shape) for name, shape in shapes.items()},
                "params": params,
            }
        )
    state = config._evaluation
    if config.getoption("list_cases"):
        emit(
            config,
            {
                "schema_version": "flaggems.benchmark-case-list/v2",
                "benchmarks": [
                    {
                        "op_name": op_name,
                        "level": "core",
                        "phase": "timing",
                        "cases": rows,
                    }
                ],
            },
        )
        return

    selected = set(config.getoption("case_id"))
    known = {row["case_id"] for row in rows}
    if selected - known:
        raise pytest.UsageError(
            f"Unknown --case-id: {', '.join(sorted(selected - known))}"
        )
    candidate = request.getfixturevalue("candidate")
    device = config.getoption("device")
    baseline_path = config.getoption("base_data")
    baseline = (
        json.loads(baseline_path.read_text())
        if baseline_path
        and candidate.source != "dummy"
        and not config.getoption("profile_only")
        else None
    )
    workloads = [
        (case, row)
        for case, row in zip(cases, rows)
        if not selected or row["case_id"] in selected
    ]
    probe_start = len(state.records)
    by_dtype = {}
    for case, row in workloads:
        case_id = row["case_id"]
        build = partial(make_inputs, case, device)
        if config.getoption("profile_only"):
            _profile(config, case_id, candidate, build)
            return
        if candidate.source == "dummy":
            state.run_probe(candidate, build, request.node.nodeid, case_id)
            continue

        args = build()
        before = state.calls
        latency = _measure_ms(candidate, args)
        assert state.calls > before, "Candidate timer did not execute the candidate"
        assert math.isfinite(latency) and latency > 0
        state.timings += 1
        base_latency_ms = None
        miss_reason = None
        if baseline is not None:
            match_shape = row["shape"][baseline_tensor]
            base_us, miss_reason = _lookup_baseline(baseline, op_name, match_shape, row)
            if base_us is not None:
                base_latency_ms = base_us / 1000
        result = {
            "case_id": case_id,
            "shape_detail": row["shape"],
            "params": row["params"],
            "latency": latency,
            "candidate_source": candidate.source,
            "base_latency_ms": base_latency_ms,
            "base_raw_speedup": (
                base_latency_ms / latency if base_latency_ms is not None else None
            ),
            "compared_speedup": None,
            "base_miss_reason": miss_reason,
        }
        by_dtype.setdefault(row["dtype"], []).append(result)

    if candidate.source == "dummy":
        expected = [row["case_id"] for _, row in workloads]
        observed = [record["case_id"] for record in state.records[probe_start:]]
        assert observed == expected, "Dummy records differ from the selected workloads"
        return

    details = [
        {"op_name": op_name, "dtype": dtype, "level": "core", "result": results}
        for dtype, results in by_dtype.items()
    ]
    emit(
        config,
        {
            op_name: {
                "details": details,
                "result": "passed",
                "test_case": request.node.nodeid,
            }
        },
    )
