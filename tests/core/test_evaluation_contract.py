"""Exercise the public pytest interface in isolated subprocesses."""

import json
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def suite(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    pytester.makeini("[pytest]\naddopts = --device cpu")
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


def test_reference_rejects_missing_declaration(suite):
    with (suite.path / "conftest.py").open("a") as out:
        out.write("\n_DEFAULT_REFERENCES.clear()\n")
    suite.makepyfile("def test_probe(candidate): candidate(1)")
    result = suite.runpytest_subprocess("--ref-as-candidate")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*No predefined reference*"])


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
@pytest.fixture(autouse=True)
def fake_timer(monkeypatch):
    monkeypatch.setattr(__import__(__name__, fromlist=[""]), "make_inputs", lambda *a: (1,))
    monkeypatch.setattr(__import__(__name__, fromlist=[""]), "measure_us", lambda *a: 1.0)
"""
    )
    result = suite.runpytest_subprocess(
        "benchmark/test_add_benchmark.py", "--ref-as-candidate"
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Candidate timer did not execute the candidate*"])


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


def test_dummy_benchmark_continues_after_skipped_input(suite):
    directory = suite.path / "benchmark"
    directory.mkdir()
    (directory / "test_add_benchmark.py").write_text(
        (ROOT / "benchmark/test_add_benchmark.py").read_text() + """
import sys
@pytest.fixture(autouse=True)
def fake_inputs(monkeypatch):
    def inputs(case, device):
        if case.dtype == "uint8":
            pytest.skip("dtype unavailable")
        return (1,)
    monkeypatch.setattr(sys.modules[__name__], "make_inputs", inputs)
"""
    )
    suite.runpytest_subprocess(
        "benchmark/test_add_benchmark.py", "--dummy-candidate", "add"
    ).assert_outcomes(passed=1)
    report = json.loads((suite.path / "dummy-candidate-report.json").read_text())
    assert report["status"] == "PASSED"
    assert len(report["records"]) == 8
    assert [row["status"] for row in report["records"]] == ["REACHED", "SKIP"] * 4


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
