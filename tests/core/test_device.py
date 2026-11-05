"""Device selection uses one explicit vendor contract across evaluation."""

from types import SimpleNamespace

import pytest

from runtime import device as runtime


@pytest.fixture(autouse=True)
def clean_vendor(monkeypatch):
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.delenv("PPU_SDK", raising=False)


@pytest.mark.parametrize(
    "vendor, device",
    [
        ("nvidia", "cuda"),
        ("ascend", "npu"),
        ("mthreads", "musa"),
        ("cambricon", "mlu"),
        ("enflame", "gcu"),
        ("thead", "cuda"),
    ],
)
def test_explicit_vendor(monkeypatch, vendor, device):
    monkeypatch.setenv("GEMS_VENDOR", vendor.upper())
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: None)
    assert runtime.resolve_device() == runtime.RuntimeDevice(vendor, device)


def test_invalid_vendor_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("GEMS_VENDOR", "invalid")
    with pytest.raises(ValueError, match="Unknown GEMS_VENDOR"):
        runtime.resolve_device()


def test_explicit_cpu_needs_no_torch(monkeypatch):
    monkeypatch.setattr(
        runtime, "_torch_runtime", lambda: pytest.fail("imported Torch")
    )
    assert runtime.resolve_device("cpu") == runtime.RuntimeDevice("cpu", "cpu")


def test_device_index_is_preserved(monkeypatch):
    monkeypatch.setenv("GEMS_VENDOR", "ascend")
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: None)
    assert runtime.resolve_device("npu:1").device == "npu:1"
    with pytest.raises(ValueError, match="incompatible"):
        runtime.resolve_device("cuda:0")


def test_automatic_vendor_is_used(monkeypatch):
    monkeypatch.setattr(runtime, "_torch_runtime", lambda: None)
    monkeypatch.setattr(runtime, "_detect_vendor", lambda torch: "mthreads")
    assert runtime.resolve_device() == runtime.RuntimeDevice("mthreads", "musa")


def test_detects_nvidia_runtime():
    torch = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(name="NVIDIA H800"),
        ),
    )
    assert runtime._detect_vendor(torch) == "nvidia"


def test_detection_failure_is_explicit(monkeypatch):
    torch = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1)
    )
    with pytest.raises(RuntimeError, match="No accelerator detected"):
        runtime._detect_vendor(torch)
