"""Opt-in pytest entry points for FlagCV kernel evaluation."""

import importlib.util
import json
from pathlib import Path

import pytest

from evaluation.reference import get_reference


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


def pytest_addoption(parser):
    group = parser.getgroup("FlagCV evaluation")
    group.addoption("--device", default="cuda", help="Torch device for evaluation")
    group.addoption("--level", choices=["core"], default="core")
    group.addoption("--override", metavar="add:FILE[:FUNCTION]")
    group.addoption("--dummy-candidate", choices=["add"])
    group.addoption("--ref-as-candidate", action="store_true")
    group.addoption("--list-cases", action="store_true")
    group.addoption("--case-id", action="append", default=[])
    group.addoption("--output", type=Path, help="JSON catalog or benchmark result path")
    group.addoption("--base-data", type=Path, help="ops-baseline OpenCV CUDA JSON")
    group.addoption(
        "--reference", metavar="FILE:FUNCTION", help="Independent correctness oracle"
    )


def pytest_configure(config):
    modes = ("override", "dummy_candidate", "ref_as_candidate", "list_cases")
    if sum(bool(config.getoption(mode)) for mode in modes) > 1:
        raise pytest.UsageError(
            "Choose one of --override, --dummy-candidate, "
            "--ref-as-candidate, or --list-cases"
        )
    selected = config.getoption("case_id")
    if selected and config.getoption("list_cases"):
        raise pytest.UsageError("--list-cases enumerates all cases; omit --case-id")


class DummyReached(Exception):
    """Signal raised only by the injected probe."""


def dummy(*args):
    raise DummyReached


@pytest.fixture(scope="session")
def candidate(request):
    config = request.config
    if config.getoption("dummy_candidate"):
        return dummy
    if config.getoption("ref_as_candidate"):
        return get_reference("add", config.getoption("device"))
    override = config.getoption("override")
    if override:
        parts = override.split(":")
        if len(parts) not in (2, 3) or parts[0] != "add" or not all(parts):
            raise pytest.UsageError("Expected --override add:FILE[:FUNCTION]")
        return _load_callable(parts[1], parts[2] if len(parts) == 3 else "add")
    pytest.skip(
        "Select a candidate with --override, --dummy-candidate or --ref-as-candidate"
    )


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
    payload = json.dumps(data, indent=2)
    path = config.getoption("output")
    if path is None:
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(payload)
    else:
        path.write_text(payload + "\n", encoding="utf-8")
