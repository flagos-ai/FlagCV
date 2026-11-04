"""Opt-in pytest entry points for FlagCV kernel evaluation."""

import json
from pathlib import Path

import pytest

from evaluation.add import load_callable, reference


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
    group.addoption("--opencv-recipe", type=Path, help="ops-baseline ops/opencv/add.py")


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
        return reference
    override = config.getoption("override")
    if override:
        parts = override.split(":")
        if len(parts) not in (2, 3) or parts[0] != "add" or not all(parts):
            raise pytest.UsageError("Expected --override add:FILE[:FUNCTION]")
        return load_callable(parts[1], parts[2] if len(parts) == 3 else "add")
    pytest.skip(
        "Select a candidate with --override, --dummy-candidate or --ref-as-candidate"
    )


@pytest.fixture(scope="session")
def native_reference(request):
    path = request.config.getoption("opencv_recipe")
    if path is None:
        return None
    native = load_callable(path, "native")()
    if native is None:
        pytest.fail("OpenCV CUDA recipe could not load its native bridge")
    return native


def emit(config, data):
    payload = json.dumps(data, indent=2)
    path = config.getoption("output")
    if path is None:
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(payload)
    else:
        path.write_text(payload + "\n", encoding="utf-8")
