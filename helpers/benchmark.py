"""Run, list and profile FlagCV benchmarks."""

import math
from contextlib import nullcontext
from functools import partial
from statistics import median

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


def _prepare_baseline(op_name, args, vendor):
    import torch

    if vendor != "nvidia":
        pytest.fail(
            "OpenCV CUDA baseline requires NVIDIA; use --baseline none for timing only"
        )
    from adapters.opencv_cuda import prepare

    dst_index = 1 if op_name in ("bitwise_not", "flip") else 2
    baseline_args = list(args)
    baseline_args[dst_index] = torch.empty_like(args[dst_index])
    baseline_args = tuple(baseline_args)
    return prepare(op_name, baseline_args), baseline_args


def _validate_outputs(op_name, candidate, args, baseline, baseline_args):
    """Validate both implementations before timing on identical inputs."""
    import torch

    from helpers.correctness import poison_output
    from helpers.tolerance import (
        assert_bitwise_equal,
        assert_exact,
        assert_one_lsb,
        assert_relative_close,
    )
    from reference import get_reference

    dst_index = 1 if op_name in ("bitwise_not", "flip") else 2
    sources, dst, extra = args[:dst_index], args[dst_index], args[dst_index + 1 :]
    snapshots = tuple(source.clone() for source in sources)
    expected = torch.empty_like(dst)
    reference_args = tuple(source.clone() for source in snapshots)
    get_reference(op_name, dst.device)(*reference_args, expected, *extra)
    if op_name == "flip":
        compare = assert_bitwise_equal
    elif dst.dtype == torch.uint8:
        compare = assert_one_lsb if op_name in ("add", "subtract") else assert_exact
    else:
        compare = assert_relative_close

    scope = (
        torch.get_device_module(dst.device).device(dst.device)
        if dst.device.type != "cpu"
        else nullcontext()
    )
    with scope:
        for function, operands in ((baseline, baseline_args), (candidate, args)):
            if function is None:
                continue
            poison_output(operands[dst_index], expected)
            assert (
                function(*operands) is operands[dst_index]
            ), "Operator must return preallocated dst"
            compare(operands[dst_index], expected)
            for source, snapshot in zip(sources, snapshots):
                assert_bitwise_equal(source, snapshot)
    if baseline is not None:
        compare(dst, baseline_args[dst_index])


def _time_case(config, op_name, candidate, args):
    baseline = None
    baseline_args = None
    if config.getoption("baseline") == "opencv-cuda":
        baseline, baseline_args = _prepare_baseline(
            op_name, args, config._flagcv_runtime.vendor
        )
    try:
        _validate_outputs(op_name, candidate, args, baseline, baseline_args)
        candidate_samples, baseline_samples = [], []
        for sample in range(config.getoption("benchmark_samples")):
            order = (False, True) if sample % 2 == 0 else (True, False)
            for is_baseline in order:
                if is_baseline and baseline is None:
                    continue
                function = baseline if is_baseline else candidate
                operands = baseline_args if is_baseline else args
                calls = 0

                def observed(*values):
                    nonlocal calls
                    calls += 1
                    return function(*values)

                latency = _measure_ms(observed, operands)
                assert calls > 0, "Candidate timer did not execute the candidate"
                assert math.isfinite(latency) and latency > 0
                (baseline_samples if is_baseline else candidate_samples).append(latency)
        config._evaluation.timings += 1
        latency = median(candidate_samples)
        base_latency = median(baseline_samples) if baseline_samples else None
        speedup = base_latency / latency if base_latency is not None else None
        return {
            "latency": latency,
            "latency_samples_ms": candidate_samples,
            "base_latency_ms": base_latency,
            "base_latency_samples_ms": baseline_samples,
            "base_raw_speedup": speedup,
            "compared_speedup": speedup,
            "baseline_source": "opencv_cuda_live" if baseline else None,
            "baseline_version": baseline.version if baseline else None,
            "base_miss_reason": None if baseline else "baseline disabled",
            "accuracy_checked": True,
        }
    finally:
        if baseline is not None:
            baseline.close()


def run(request, config, *, op_name, cases, make_inputs, describe):
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
        timing = _time_case(config, op_name, candidate, args)
        result = {
            "case_id": case_id,
            "shape_detail": row["shape"],
            "params": row["params"],
            "candidate_source": candidate.source,
            **timing,
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
