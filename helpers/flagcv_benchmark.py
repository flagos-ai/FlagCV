"""FlagCV runner for the five elementwise benchmark profiles."""

import math
from contextlib import nullcontext
from functools import partial

import pytest

from conftest import emit
from workloads.elementwise import CASES, make_inputs


def _measure_ms(candidate, args):
    import torch
    from triton.testing import do_bench

    device = args[0].device
    if device.type == "cpu":
        pytest.fail("Device timing requires an accelerator")
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
        assert config._evaluation.calls > before


def run(op_name, request, config):
    workloads = CASES[op_name]
    rows = [
        case.describe(op_name, request.node.nodeid, index)
        for index, case in enumerate(workloads)
    ]
    state = config._evaluation
    if config.getoption("list_cases"):
        state.payload.setdefault("schema_version", "flaggems.benchmark-case-list/v2")
        state.payload.setdefault("benchmarks", []).append(
            {"op_name": op_name, "level": "core", "phase": "timing", "cases": rows}
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
    by_dtype = {}
    for case, row in zip(workloads, rows):
        case_id = row["case_id"]
        if selected and case_id not in selected:
            continue
        build = partial(make_inputs, op_name, case, device)
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
        by_dtype.setdefault(row["dtype"], []).append(
            {
                "case_id": case_id,
                "shape_detail": row["shape"],
                "params": row["params"],
                "latency": latency,
                "candidate_source": candidate.source,
            }
        )

    if candidate.source != "dummy":
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
