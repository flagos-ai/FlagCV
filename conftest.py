"""Shared pytest contract for FlagCV correctness and benchmark evaluation."""

import importlib.util
import json
from pathlib import Path

import pytest

from reference import get_reference
from runtime import resolve_device

OPERATORS = tuple(
    sorted(path.stem for path in Path(__file__).with_name("definitions").glob("*.json"))
)


def _test_operator(item):
    matches = {mark.name for mark in item.iter_markers()} & set(OPERATORS)
    if len(matches) > 1:
        raise pytest.UsageError(f"Multiple operator markers on {item.nodeid}")
    if matches:
        return next(iter(matches))
    if len(OPERATORS) == 1:
        return OPERATORS[0]
    return None


class DummyReached(Exception):
    """A candidate invocation probe, carrying no numerical result."""


class ProfileHooks:
    @pytest.hookspec(firstresult=True)
    def pytest_flaggems_profile_scope(self, backend, case_id):
        """Return a capture context after preparation and warmup."""


def pytest_addhooks(pluginmanager):
    pluginmanager.add_hookspecs(ProfileHooks)


def pytest_addoption(parser):
    group = parser.getgroup("FlagCV evaluation")
    group.addoption(
        "--device",
        default=None,
        help="Optional device override; otherwise detect via GEMS_VENDOR/runtime",
    )
    group.addoption("--level", choices=["core"], default="core")
    group.addoption(
        "--override", action="append", default=[], metavar="OP:FILE[:FUNCTION]"
    )
    group.addoption("--dummy-candidate", choices=OPERATORS)
    group.addoption("--dummy-report", type=Path)
    group.addoption("--ref-as-candidate", action="store_true")
    group.addoption("--list-cases", action="store_true")
    group.addoption("--profile-only", action="store_true")
    group.addoption("--profile-warmup", type=int, default=10)
    group.addoption("--profile-iterations", type=int, default=1)
    group.addoption("--case-id", action="append", default=[])
    group.addoption("--record", choices=["none", "log", "json"], default="none")
    group.addoption("--output", type=Path)
    group.addoption("--base-data", type=Path)
    group.addoption("--reference", metavar="FILE:FUNCTION", help="Independent oracle")


def pytest_configure(config):
    def enabled(name):
        return bool(config.getoption(name))

    modes = ("override", "dummy_candidate", "ref_as_candidate", "list_cases")
    if sum(enabled(mode) for mode in modes) > 1:
        raise pytest.UsageError(
            "Choose one of --override, --dummy-candidate, "
            "--ref-as-candidate or --list-cases"
        )
    if enabled("dummy_report") and not enabled("dummy_candidate"):
        raise pytest.UsageError("--dummy-report requires --dummy-candidate")
    selected = config.getoption("case_id")
    if len(selected) != len(set(selected)):
        raise pytest.UsageError("Duplicate --case-id values are not allowed")
    if enabled("list_cases") and selected:
        raise pytest.UsageError("--list-cases cannot be combined with --case-id")
    if enabled("profile_only"):
        if any(
            enabled(name)
            for name in ("list_cases", "dummy_candidate", "ref_as_candidate")
        ):
            raise pytest.UsageError(
                "--profile-only cannot be combined with --list-cases, "
                "--dummy-candidate or --ref-as-candidate"
            )
        if len(selected) != 1:
            raise pytest.UsageError("--profile-only requires exactly one --case-id")
        if not enabled("override"):
            raise pytest.UsageError("--profile-only requires --override")
    if (
        config.getoption("profile_warmup") < 0
        or config.getoption("profile_iterations") < 1
    ):
        raise pytest.UsageError(
            "profile warmup must be non-negative and iterations positive"
        )
    if any(
        enabled(name)
        for name in ("dummy_candidate", "ref_as_candidate", "profile_only")
    ):
        if config.option.collectonly or getattr(config.option, "numprocesses", None):
            raise pytest.UsageError(
                "Candidate validation requires execution without xdist"
            )
    for operator in OPERATORS:
        config.addinivalue_line("markers", f"{operator}: FlagCV operator evaluation")
    config._evaluation = Evaluation(config)
    config.pluginmanager.register(config._evaluation, "flagcv-evaluation")


def _load_callable(path, name):
    spec = importlib.util.spec_from_file_location("_flagcv_evaluation_external", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load Python module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, name, None)
    if not callable(function):
        raise ValueError(f"{path!r} does not define callable {name!r}")
    return function


class Evaluation:
    def __init__(self, config):
        self.config = config
        self.calls = 0
        self.timings = 0
        self.active = None
        self.records = []
        self.results = {}
        self.payload = {}
        self.collection_errors = []

    def observe(self, function, source):
        def invoke(*args, **kwargs):
            self.calls += 1
            return function(*args, **kwargs)

        invoke.source = source
        return invoke

    def probe(self):
        def invoke(*args):
            self.calls += 1
            error = DummyReached("DUMMY_CANDIDATE_REACHED")
            error.owner = self
            raise error

        invoke.source = "dummy"
        return invoke

    def run_probe(self, candidate, build_inputs, nodeid, case_id=None):
        before = self.calls
        record = {
            "operator": self.config.getoption("dummy_candidate"),
            "phase": "benchmark" if case_id else "correctness",
            "nodeid": nodeid,
            "status": "PRE_CALL_ERROR",
            "calls": 0,
            "before_candidate_calls": 0,
            "unexpected_calls": 0,
        }
        if case_id:
            record["case_id"] = case_id
        self.records.append(record)
        invoking = False
        try:
            inputs = build_inputs()
            if self.calls != before:
                pytest.fail("Candidate invoked during input preparation")
            invoking = True
            candidate(*inputs)
        except DummyReached as error:
            if (
                not invoking
                or getattr(error, "owner", None) is not self
                or self.calls <= before
            ):
                record["message"] = "Unobserved dummy signal"
                raise
            record["status"] = "REACHED"
        except BaseException as error:
            if (
                isinstance(error, pytest.skip.Exception)
                and not invoking
                and self.calls == before
            ):
                record["status"] = "SKIP"
                record["message"] = str(error)
                if case_id is not None:
                    return
            elif invoking:
                record["status"] = "POST_CALL_ERROR"
            record["message"] = str(error)
            raise
        else:
            record["status"] = "BYPASSED"
            pytest.fail("Selected case did not invoke the dummy candidate")
        finally:
            record["calls"] = self.calls - before

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_call(self, item):
        self.calls = 0
        self.timings = 0
        self.active = item.nodeid
        try:
            yield
        finally:
            self.active = None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = (yield).get_result()
        if (
            self.config.getoption("ref_as_candidate")
            and report.when in ("setup", "call")
            and not report.failed
            and (report.when == "call" or report.skipped)
            and not getattr(item, "_reference_declared", False)
        ):
            report.outcome = "failed"
            report.longrepr = "No predefined reference selected for this test"
            if hasattr(report, "wasxfail"):
                del report.wasxfail
        dummy_mode = self.config.getoption("dummy_candidate")
        if (
            dummy_mode
            and (report.failed or report.skipped)
            and not any(record["nodeid"] == item.nodeid for record in self.records)
        ):
            self.records.append(
                {
                    "operator": dummy_mode,
                    "phase": (
                        "benchmark"
                        if (Path(__file__).resolve().parent / "benchmark")
                        in item.path.parents
                        else "correctness"
                    ),
                    "nodeid": item.nodeid,
                    "status": "SKIP" if report.skipped else "PRE_CALL_ERROR",
                    "calls": 0,
                    "before_candidate_calls": 0,
                    "unexpected_calls": 0,
                    "message": str(report.longrepr),
                }
            )
        if (
            dummy_mode
            and report.skipped
            and any(
                record["nodeid"] == item.nodeid and record["status"] == "REACHED"
                for record in self.records
            )
        ):
            report.outcome = "failed"
            report.longrepr = "Dummy candidate signal was followed by skip or xfail"
            if hasattr(report, "wasxfail"):
                del report.wasxfail
            self.records.append(
                {
                    "operator": dummy_mode,
                    "phase": (
                        "benchmark"
                        if (Path(__file__).resolve().parent / "benchmark")
                        in item.path.parents
                        else "correctness"
                    ),
                    "nodeid": item.nodeid,
                    "status": "POST_CALL_ERROR",
                    "calls": 0,
                    "before_candidate_calls": 0,
                    "unexpected_calls": 0,
                    "message": str(report.longrepr),
                }
            )
        if report.when != "call":
            if report.failed or report.skipped:
                self.results[item.nodeid] = {
                    "result": report.outcome,
                    "reason": str(report.longrepr),
                }
            return
        validating = any(
            self.config.getoption(name)
            for name in (
                "override",
                "dummy_candidate",
                "ref_as_candidate",
                "profile_only",
            )
        )
        requires_timing = (
            self.config.getoption("ref_as_candidate")
            and (Path(__file__).resolve().parent / "benchmark") in item.path.parents
        )
        if (
            validating
            and report.passed
            and (not self.calls or (requires_timing and not self.timings))
        ):
            report.outcome = "failed"
            report.longrepr = "Selected test did not complete its candidate execution"
        if report.failed:
            for record in self.records:
                if (
                    record["nodeid"] == item.nodeid
                    and record["status"] == "REACHED"
                    and record["phase"] == "correctness"
                ):
                    record["status"] = "POST_CALL_ERROR"
        operator = getattr(item, "_candidate_operator", None)
        if operator is None and len(OPERATORS) == 1:
            operator = OPERATORS[0]
        result = {
            "result": report.outcome,
            "reason": str(report.longrepr) if report.failed or report.skipped else None,
            "candidate_calls": {f"flagcv.{operator}": self.calls} if operator else {},
        }
        if self.config.getoption("ref_as_candidate"):
            result.update(
                candidate_source="reference",
                reference_calls=self.calls,
                reference_candidate_calls=self.calls,
            )
        self.results[item.nodeid] = result

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        config = self.config
        if config.getoption("dummy_candidate"):
            successful = (
                any(record["status"] == "REACHED" for record in self.records)
                and all(
                    record["status"] in {"REACHED", "SKIP"} for record in self.records
                )
                and session.exitstatus == 0
            )
            status = "PASSED" if successful else "FAILED"
            if not self.records and session.exitstatus in (0, 5):
                status = "NO_CASES"
            elif self.records and all(
                record["status"] == "SKIP" for record in self.records
            ):
                status = "ALL_SKIP"
            if not successful and session.exitstatus in (0, 5):
                session.exitstatus = pytest.ExitCode.TESTS_FAILED
            write_json(
                config.getoption("dummy_report") or Path("dummy-candidate-report.json"),
                {
                    "schema_version": "flaggems.dummy-candidate/v1",
                    "operator": config.getoption("dummy_candidate"),
                    "status": status,
                    "proves": "candidate_invocation_only",
                    "records": self.records,
                    "collection_errors": self.collection_errors,
                },
            )
            return
        if config.getoption("ref_as_candidate") and not any(
            row.get("reference_calls", 0) and row["result"] == "passed"
            for row in self.results.values()
        ):
            if session.exitstatus in (0, 5):
                session.exitstatus = pytest.ExitCode.TESTS_FAILED
        if config.getoption("list_cases"):
            if not self.payload and session.exitstatus in (0, 5):
                session.exitstatus = pytest.ExitCode.TESTS_FAILED
            write_json(
                config.getoption("output") or Path("benchmark_result.json"),
                self.payload,
            )
        elif config.getoption("record") == "json":
            data = dict(self.results)
            data.update(self.payload)
            write_json(
                config.getoption("output")
                or Path(
                    "benchmark_result.json" if self.payload else "accuracy_result.json"
                ),
                data,
            )
        elif config.getoption("record") == "log":
            reporter = config.pluginmanager.get_plugin("terminalreporter")
            if reporter:
                reporter.write_line(json.dumps(self.payload or self.results))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def _candidate_runtime(pytestconfig):
    runtime = resolve_device(pytestconfig.getoption("device"))
    pytestconfig._flagcv_runtime = runtime
    pytestconfig.option.device = runtime.device
    return runtime


@pytest.fixture(scope="session")
def _override_candidates(pytestconfig):
    functions = {}
    for override in pytestconfig.getoption("override"):
        parts = override.replace("=", ":", 1).split(":")
        if len(parts) not in (2, 3) or parts[0] not in OPERATORS or not all(parts):
            raise pytest.UsageError("Expected --override OP:FILE[:FUNCTION]")
        operator, path = parts[:2]
        functions[operator] = _load_callable(
            path, parts[2] if len(parts) == 3 else operator
        )
    return functions


@pytest.fixture
def candidate(request):
    config = request.config
    if not any(
        config.getoption(name)
        for name in ("dummy_candidate", "ref_as_candidate", "override")
    ):
        pytest.skip(
            "Select a candidate with --override, --dummy-candidate "
            "or --ref-as-candidate"
        )
    operator = _test_operator(request.node)
    if operator is None:
        raise pytest.UsageError(f"No operator marker on {request.node.nodeid}")
    request.node._candidate_operator = operator
    runtime = request.getfixturevalue("_candidate_runtime")
    state = config._evaluation
    if config.getoption("dummy_candidate"):
        return state.probe()
    if config.getoption("ref_as_candidate"):
        reference = get_reference(operator, runtime.device)
        request.node._reference_declared = True
        return state.observe(reference, "reference")
    function = request.getfixturevalue("_override_candidates").get(operator)
    if function is None:
        raise pytest.UsageError(f"No --override supplied for {operator!r}")
    return state.observe(function, "override")


@pytest.fixture(scope="session")
def native_reference(request):
    spec = request.config.getoption("reference")
    if spec is None:
        return None
    path, sep, function = spec.rpartition(":")
    if not sep or not path or not function:
        raise pytest.UsageError("Expected --reference FILE:FUNCTION")
    return _load_callable(path, function)


def emit(config, data):
    config._evaluation.payload.update(data)


def pytest_collection_modifyitems(config, items):
    dummy_operator = config.getoption("dummy_candidate")
    if dummy_operator:
        selected = [item for item in items if _test_operator(item) == dummy_operator]
    elif config.getoption("profile_only"):
        benchmark_dir = Path(__file__).resolve().parent / "benchmark"
        selected = [item for item in items if benchmark_dir in item.path.parents]
        if not selected:
            raise pytest.UsageError("--profile-only requires a benchmark test")
    else:
        return
    deselected = [item for item in items if item not in selected]
    if deselected:
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
