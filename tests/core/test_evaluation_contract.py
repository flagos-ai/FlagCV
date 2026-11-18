"""Exercise the public pytest interface in isolated subprocesses."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def suite(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    pytester.makeini("[pytest]\naddopts = --device cpu")
    reference_dir = pytester.mkdir("reference")
    for name in ("__init__.py", "add.py"):
        (reference_dir / name).write_text((ROOT / "reference" / name).read_text())
    pytester.makeconftest(
        (ROOT / "conftest.py").read_text()
        + "\nfrom reference import _DEFAULT_REFERENCES\n"
        + '_DEFAULT_REFERENCES["add"] = lambda value: value + 1\n'
    )
    return pytester


def test_reference_mode_observes_each_case(suite):
    suite.makepyfile("""
import pytest
@pytest.mark.parametrize("use_candidate", [True, False])
def test_probe(candidate, use_candidate):
    if use_candidate:
        assert candidate(1) == 2
""")
    suite.runpytest_subprocess(
        "--ref-as-candidate", "--record", "json", "--output", "out.json"
    ).assert_outcomes(passed=1, failed=1)
    rows = list(json.loads((suite.path / "out.json").read_text()).values())
    assert [row["reference_candidate_calls"] for row in rows] == [1, 0]
    assert [row["params"] for row in rows] == [
        {"use_candidate": True},
        {"use_candidate": False},
    ]
    assert [row["opname"] for row in rows] == [["add"], ["add"]]


def test_add_reports_shape_and_dtype(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    output = tmp_path / "add-accuracy.json"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_add.py",
            "--device",
            "cpu",
            "--ref-as-candidate",
            "--record",
            "json",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    rows = json.loads(output.read_text())
    expected = {
        ((64, 64), "float32"),
        ((64, 64), "uint8"),
        ((480, 640), "float32"),
        ((480, 640), "uint8"),
        ((1080, 1920), "float32"),
        ((1080, 1920), "uint8"),
        ((2160, 3840), "float32"),
        ((2160, 3840), "uint8"),
        ((1, 3), "uint8"),
    }
    assert len(rows) == len(expected)
    assert {
        (tuple(row["params"]["shape"]), row["params"]["dtype"])
        for row in rows.values()
    } == expected
    for row in rows.values():
        assert isinstance(row["params"]["shape"], list)
        assert row["opname"] == ["add"]
        assert row["result"] == "passed"


def test_reference_rejects_missing_declaration(suite):
    (suite.path / "reference/missing.py").write_text("# No reference callable.\n")
    suite.makepyfile("""
import pytest
@pytest.mark.missing
def test_probe(candidate):
    candidate(1)
""")
    result = suite.runpytest_subprocess(
        "--ref-as-candidate", "--record", "json", "--output", "out.json"
    )
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*No predefined reference*"])
    (row,) = json.loads((suite.path / "out.json").read_text()).values()
    assert row["opname"] == ["missing"]
    assert row["params"] == {}


@pytest.fixture
def two_operators(suite):
    reference_dir = suite.path / "reference"
    (reference_dir / "subtract.py").write_text(
        "def reference(value): return value - 1\n"
    )
    suite.makepyfile("""
import pytest

@pytest.mark.add
def test_add(candidate):
    if candidate.source == "dummy":
        pytest.fail("Add should be deselected when probing subtract")
    assert candidate(4) == 5

@pytest.mark.subtract
def test_subtract(candidate, pytestconfig, request):
    if candidate.source == "dummy":
        pytestconfig._evaluation.run_probe(candidate, lambda: (4,), request.node.nodeid)
    else:
        assert candidate(4) == 3
""")
    return suite


@pytest.mark.parametrize("mode", ["reference", "override"])
def test_candidate_dispatches_by_test_operator(two_operators, mode):
    suite = two_operators
    if mode == "reference":
        args = ["--ref-as-candidate"]
    else:
        suite.makepyfile(implementations="""
def add(value): return value + 1
def subtract(value): return value - 1
""")
        args = [
            "--override", "add:implementations.py",
            "--override", "subtract:implementations.py",
        ]
    suite.runpytest_subprocess(
        *args, "--record", "json", "--output", "out.json"
    ).assert_outcomes(passed=2)
    rows = json.loads((suite.path / "out.json").read_text())
    calls = {
        nodeid.rsplit("::", 1)[-1]: row["candidate_calls"]
        for nodeid, row in rows.items()
    }
    assert calls == {
        "test_add": {"flagcv.add": 1},
        "test_subtract": {"flagcv.subtract": 1},
    }


def test_dummy_selects_only_requested_operator(two_operators):
    suite = two_operators
    suite.runpytest_subprocess(
        "--dummy-candidate", "subtract", "--dummy-report", "dummy.json"
    ).assert_outcomes(passed=1, deselected=1)
    report = json.loads((suite.path / "dummy.json").read_text())
    assert report["status"] == "PASSED"
    (record,) = report["records"]
    assert record["operator"] == "subtract"
    assert record["nodeid"].endswith("::test_subtract")
    assert record["status"] == "REACHED"
    assert record["calls"] == 1


def test_dummy_reports_individual_calls(suite):
    suite.makepyfile("""
def test_probe(candidate, pytestconfig, request):
    pytestconfig._evaluation.run_probe(candidate, lambda: (1,), request.node.nodeid)
""")
    suite.runpytest_subprocess(
        "--dummy-candidate", "add", "--dummy-report", "dummy.json"
    ).assert_outcomes(passed=1)
    report = json.loads((suite.path / "dummy.json").read_text())
    assert report["schema_version"] == "flaggems.dummy-candidate/v1"
    assert report["status"] == "PASSED"
    assert report["records"][0]["calls"] == 1


def test_dummy_bypass_fails(suite):
    suite.makepyfile("def test_bypass(candidate): pass")
    suite.runpytest_subprocess("--dummy-candidate", "add").assert_outcomes(failed=1)
    report = json.loads((suite.path / "dummy-candidate-report.json").read_text())
    assert report["status"] == "FAILED"


@pytest.mark.parametrize(
    "args",
    [
        ["--profile-only", "--ref-as-candidate", "--case-id", "case"],
        ["--dummy-report", "unused.json"],
        ["--case-id", "case", "--case-id", "case"],
        ["--profile-iterations", "0"],
    ],
)
def test_invalid_options_are_usage_errors(suite, args):
    suite.makepyfile("def test_probe(): pass")
    assert suite.runpytest_subprocess(*args).ret == pytest.ExitCode.USAGE_ERROR


def test_override_accepts_equals_and_tracks_calls(suite):
    suite.makepyfile(candidate="def add(value): return value + 2")
    suite.makepyfile(test_probe="def test_probe(candidate): assert candidate(1) == 3")
    suite.runpytest_subprocess(
        "test_probe.py",
        "--override",
        "add=candidate.py",
        "--record",
        "json",
        "--output",
        "out.json",
    ).assert_outcomes(passed=1)
    row = next(iter(json.loads((suite.path / "out.json").read_text()).values()))
    assert row["candidate_calls"] == {"flagcv.add": 1}


def test_dummy_signal_during_preparation_is_not_a_hit(suite):
    suite.makepyfile("""
def test_probe(candidate, pytestconfig, request):
    pytestconfig._evaluation.run_probe(
        candidate, lambda: candidate(1), request.node.nodeid
    )
""")
    suite.runpytest_subprocess("--dummy-candidate", "add").assert_outcomes(failed=1)
    report = json.loads((suite.path / "dummy-candidate-report.json").read_text())
    assert report["records"][0]["status"] == "PRE_CALL_ERROR"


def test_benchmark_rejects_timer_that_bypasses_candidate(suite):
    directory = suite.path / "benchmark"
    directory.mkdir()
    (directory / "test_add_benchmark.py").write_text(
        (ROOT / "benchmark/test_add_benchmark.py").read_text() + """
import pytest
from helpers import benchmark

@pytest.fixture(autouse=True)
def fake_timer(monkeypatch):
    module = __import__(__name__, fromlist=[""])
    monkeypatch.setattr(module, "make_inputs", lambda *a: (1,))
    monkeypatch.setattr(benchmark, "_measure_ms", lambda *a: 1.0)
"""
    )
    result = suite.runpytest_subprocess(
        "benchmark/test_add_benchmark.py", "--ref-as-candidate"
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Candidate timer did not execute the candidate*"])


@pytest.mark.parametrize("baseline", ["matched", "missing", "disabled"])
def test_benchmark_reports_baseline_fields_using_declared_tensor(suite, baseline):
    directory = suite.path / "benchmark"
    directory.mkdir()
    (directory / "test_add_benchmark.py").write_text(
        (ROOT / "benchmark/test_add_benchmark.py").read_text() + """
import sys
from helpers import benchmark

@pytest.fixture(autouse=True)
def fake_timing(monkeypatch):
    module = sys.modules[__name__]
    original_describe = module.describe

    def describe(case):
        dtype, shapes, params = original_describe(case)
        return dtype, {"dst": (32, 32), "a": shapes["a"], "b": shapes["b"]}, params

    def measure_ms(candidate, args):
        candidate(*args)
        return 0.002

    monkeypatch.setattr(module, "describe", describe)
    monkeypatch.setattr(module, "make_inputs", lambda *a: (1,))
    monkeypatch.setattr(benchmark, "_measure_ms", measure_ms)
"""
    )
    record = {
        "T_us": 8.0,
        "config": {
            "native_api": "cv::cuda::add",
            "channels": 1,
            "mask": None,
            "output_preallocated": True,
            "output_dtype": "torch.float32",
        },
    }
    baseline_data = (
        {"add": {"shapes": {"[64, 64]": {"torch.float32": record}}}}
        if baseline == "matched"
        else {}
    )
    (suite.path / "base.json").write_text(json.dumps(baseline_data))
    args = [] if baseline == "disabled" else ["--base-data", "base.json"]
    suite.runpytest_subprocess(
        "benchmark/test_add_benchmark.py",
        "--ref-as-candidate",
        "--case-id",
        "benchmark/test_add_benchmark.py::test_add_benchmark::core::float32::0",
        "--record",
        "json",
        "--output",
        "result.json",
        *args,
    ).assert_outcomes(passed=1)
    rows = json.loads((suite.path / "result.json").read_text())["add"]["details"]
    assert len(rows) == 1
    (case,) = rows[0]["result"]
    assert case["latency"] == pytest.approx(0.002)
    assert case["compared_speedup"] is None
    assert "baseline_us" not in case
    assert "raw_speedup" not in case
    if baseline == "matched":
        assert case["base_latency_ms"] == pytest.approx(0.008)
        assert case["base_raw_speedup"] == pytest.approx(4.0)
        assert case["base_miss_reason"] is None
    else:
        assert case["base_latency_ms"] is None
        assert case["base_raw_speedup"] is None
        expected_reason = (
            "missing operator/shape/dtype record" if baseline == "missing" else None
        )
        assert case["base_miss_reason"] == expected_reason


@pytest.mark.parametrize("candidate_fails", [False, True])
def test_profile_scope_excludes_warmup_and_cleans_up(suite, candidate_fails):
    benchmark = suite.path / "benchmark"
    benchmark.mkdir()
    (benchmark / "test_add_benchmark.py").write_text(
        (ROOT / "benchmark/test_add_benchmark.py").read_text() + """
import sys
from types import SimpleNamespace
@pytest.fixture(autouse=True)
def fake_inputs(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    module = sys.modules[__name__]
    value = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    monkeypatch.setattr(module, "make_inputs", lambda *a: (value,))
"""
    )
    suite.makepyfile(candidate=f"""
from pathlib import Path
def add(*args):
    path = Path("events.txt")
    with path.open("a") as out:
        out.write("call\\n")
    if {candidate_fails!r} and "enter" in path.read_text():
        raise RuntimeError("capture failure")
""")
    with (suite.path / "conftest.py").open("a") as out:
        out.write("""
from contextlib import contextmanager

@contextmanager
def pytest_flaggems_profile_scope(backend, case_id):
    assert backend == "cpu"
    with open("events.txt", "a") as out:
        out.write("enter\\n")
    try:
        yield
    finally:
        with open("events.txt", "a") as out:
            out.write("exit\\n")
""")
    result = suite.runpytest_subprocess(
        "benchmark/test_add_benchmark.py",
        "--device",
        "cpu",
        "--override",
        "add:candidate.py",
        "--profile-only",
        "--profile-warmup",
        "2",
        "--profile-iterations",
        "3",
        "--case-id",
        "benchmark/test_add_benchmark.py::test_add_benchmark::core::float32::0",
        "--base-data",
        "absent.json",
    )
    result.assert_outcomes(**({"failed": 1} if candidate_fails else {"passed": 1}))
    expected = (
        ["call", "call", "enter"] + ["call"] * (1 if candidate_fails else 3) + ["exit"]
    )
    assert (suite.path / "events.txt").read_text().splitlines() == expected


def test_reference_fixture_calls_do_not_satisfy_case(suite):
    suite.makepyfile("""
import pytest
@pytest.fixture
def setup_only(candidate):
    candidate(1)
def test_bypass(setup_only): pass
""")
    suite.runpytest_subprocess("--ref-as-candidate").assert_outcomes(failed=1)


def test_dummy_all_skip_is_not_validation_success(suite):
    suite.makepyfile("""
import pytest
@pytest.mark.skip(reason="unsupported")
def test_probe(candidate): candidate(1)
""")
    result = suite.runpytest_subprocess("--dummy-candidate", "add")
    result.assert_outcomes(skipped=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert (
        json.loads((suite.path / "dummy-candidate-report.json").read_text())["status"]
        == "ALL_SKIP"
    )


@pytest.mark.parametrize("outcome", ["skip", "xfail"])
def test_unadapted_reference_skip_cannot_hide_missing_declaration(suite, outcome):
    suite.makepyfile(f"""
import pytest
def test_valid(candidate):
    assert candidate(1) == 2
def test_unadapted():
    pytest.{outcome}("no reference declared")
""")
    result = suite.runpytest_subprocess("--ref-as-candidate")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*No predefined reference selected*"])


@pytest.fixture
def dummy_benchmark(suite):
    directory = suite.path / "benchmark"
    directory.mkdir()
    (directory / "test_add_benchmark.py").write_text(
        (ROOT / "benchmark/test_add_benchmark.py").read_text() + """
import sys
@pytest.fixture(autouse=True)
def fake_inputs(monkeypatch):
    def inputs(case, device):
        if case[1] == "uint8":
            pytest.skip("dtype unavailable")
        return (1,)
    monkeypatch.setattr(sys.modules[__name__], "make_inputs", inputs)
"""
    )
    return suite


def test_dummy_benchmark_continues_after_skipped_input(dummy_benchmark):
    suite = dummy_benchmark
    suite.runpytest_subprocess(
        "benchmark/test_add_benchmark.py", "--dummy-candidate", "add"
    ).assert_outcomes(passed=1)
    report = json.loads((suite.path / "dummy-candidate-report.json").read_text())
    assert report["status"] == "PASSED"
    assert len(report["records"]) == 8
    assert [row["status"] for row in report["records"]] == ["REACHED", "SKIP"] * 4


@pytest.mark.parametrize(
    ("second_case", "second_status", "exit_code"),
    [
        ("uint8::1", "SKIP", pytest.ExitCode.TESTS_FAILED),
        ("float32::2", "REACHED", pytest.ExitCode.OK),
    ],
)
def test_dummy_benchmark_requires_all_selected_case_hits(
    dummy_benchmark, second_case, second_status, exit_code
):
    suite = dummy_benchmark
    nodeid = "benchmark/test_add_benchmark.py::test_add_benchmark::core"
    selected = [f"{nodeid}::float32::0", f"{nodeid}::{second_case}"]
    result = suite.runpytest_subprocess(
        "benchmark/test_add_benchmark.py",
        "--dummy-candidate",
        "add",
        "--case-id",
        selected[0],
        "--case-id",
        selected[1],
    )
    assert result.ret == exit_code
    report = json.loads((suite.path / "dummy-candidate-report.json").read_text())
    assert report["status"] == (
        "PASSED" if exit_code == pytest.ExitCode.OK else "FAILED"
    )
    assert [
        (row["case_id"], row["status"], row["calls"]) for row in report["records"]
    ] == [
        (selected[0], "REACHED", 1),
        (selected[1], second_status, int(second_status == "REACHED")),
    ]


@pytest.mark.parametrize("outcome", ["skip", "xfail"])
def test_dummy_post_call_skip_is_a_failure(suite, outcome):
    suite.makepyfile(f"""
import pytest
def test_probe(candidate, pytestconfig, request):
    pytestconfig._evaluation.run_probe(candidate, lambda: (1,), request.node.nodeid)
    pytest.{outcome}("after invocation")
""")
    result = suite.runpytest_subprocess("--dummy-candidate", "add")
    result.assert_outcomes(failed=1)
    report = json.loads((suite.path / "dummy-candidate-report.json").read_text())
    assert report["status"] == "FAILED"
    assert any(row["status"] == "POST_CALL_ERROR" for row in report["records"])
