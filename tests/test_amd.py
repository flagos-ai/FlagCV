# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests, plus opt-in Triton -> HSACO -> HIP plugin integration."""

import base64
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "amd_provider", ROOT / "backends/amd/compiler.py"
)
PROVIDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVIDER)
PLUGIN = None
if "--plugin" in sys.argv:
    index = sys.argv.index("--plugin")
    PLUGIN = sys.argv[index + 1]
    del sys.argv[index : index + 2]


class RegistryFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for relative in ("kernels", "backends/amd/kernels"):
            directory = self.root / relative
            directory.mkdir(parents=True)
            (directory / "registry.json").write_text('{"kernels": []}')

    def register(self, platform="common", **changes):
        directory = self.root / (
            "kernels" if platform == "common" else "backends/amd/kernels"
        )
        shutil.copy(ROOT / "tests/fixtures/amd_kernel.py", directory / "test.py")
        entry = {"name": "test", "source": "test.py", "function": "masked_xor"}
        entry.update(changes)
        (directory / "registry.json").write_text(json.dumps({"kernels": [entry]}))
        return directory / "test.py"


class ProviderTests(RegistryFixture):
    def test_common_resolution(self):
        source = self.register()
        self.assertEqual(
            PROVIDER.resolve("test", root=self.root), (source, "masked_xor")
        )

    def test_platform_resolution_is_exclusive(self):
        self.register()
        self.register("amd", source="missing.py")
        with self.assertRaisesRegex(ValueError, "invalid source"):
            PROVIDER.resolve("test", root=self.root)

    def test_platform_precedes_common(self):
        self.register()
        source = self.register("amd")
        self.assertEqual(PROVIDER.resolve("test", root=self.root)[0], source)

    def test_path_traversal_is_rejected(self):
        self.register("amd", source="../../../tests/fixtures/amd_kernel.py")
        with self.assertRaisesRegex(ValueError, "invalid source"):
            PROVIDER.resolve("test", root=self.root)

    def test_missing_registration(self):
        with self.assertRaisesRegex(ValueError, "no AMD or common kernel"):
            PROVIDER.resolve("missing", root=self.root)

    def test_duplicate_registration(self):
        self.register()
        registry = self.root / "kernels/registry.json"
        content = json.loads(registry.read_text())
        content["kernels"] *= 2
        registry.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            PROVIDER.resolve("test", root=self.root)

    def test_identity_tracks_sources_and_registries(self):
        source = self.register()
        fake = types.SimpleNamespace(__version__="test", __file__=__file__)
        with patch.dict(sys.modules, triton=fake):
            before = PROVIDER.identify(root=self.root)
            self.assertEqual(before, PROVIDER.identify(root=self.root))
            source.write_text(source.read_text() + "\n# edit\n")
            after = PROVIDER.identify(root=self.root)
            self.assertNotEqual(before, after)
            registry = self.root / "backends/amd/kernels/registry.json"
            registry.write_text('{"kernels": [], "revision": 1}')
            self.assertNotEqual(after, PROVIDER.identify(root=self.root))

    def test_cli_rejects_invalid_backend(self):
        result = subprocess.run(
            [sys.executable, "-m", "flagcv_codegen.main", "--backend", "../amd"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "compiler")},
            input="{}",
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("invalid backend name", result.stderr)


class Error(C.Structure):
    _fields_ = [("message", C.c_char * 512)]


class Config(C.Structure):
    _fields_ = [
        ("grid", C.c_uint32 * 3),
        ("block", C.c_uint32 * 3),
        ("shared", C.c_uint32),
    ]


P = C.c_void_p
E = C.POINTER(Error)
Create = C.CFUNCTYPE(C.c_int, C.c_int, C.POINTER(P), E)
Destroy = C.CFUNCTYPE(C.c_int, P, E)
DeviceId = C.CFUNCTYPE(C.c_int, P, P, C.c_size_t, E)
Load = C.CFUNCTYPE(C.c_int, P, C.c_char_p, C.c_char_p, C.POINTER(P), E)
Unload = C.CFUNCTYPE(C.c_int, P, P, E)
Launch = C.CFUNCTYPE(C.c_int, P, P, C.POINTER(Config), C.POINTER(P), P, E)


class Api(C.Structure):
    _fields_ = [
        ("version", C.c_uint32),
        ("name", C.c_char_p),
        ("create", Create),
        ("destroy", Destroy),
        ("device_id", DeviceId),
        ("load", Load),
        ("unload", Unload),
        ("launch", Launch),
    ]


@unittest.skipUnless(PLUGIN, "enable FLAGCV_TEST_AMD_GPU to run on hardware")
class GpuTests(RegistryFixture):
    # Inherit the same registry fixture for real compiler and plugin checks.
    def setUp(self):
        super().setUp()
        import torch
        import triton

        self.torch = torch
        self.target = triton.runtime.driver.active.get_current_target()
        self.assertEqual(self.target.backend, "hip")
        self.library = C.CDLL(PLUGIN)
        self.library.flagcv_backend_api.restype = C.POINTER(Api)
        self.api = self.library.flagcv_backend_api().contents
        self.assertEqual(self.api.version, 1)
        self.assertEqual(self.api.name, b"amd")
        self.error = Error()
        self.context = P()
        self.ok(self.api.create(0, C.byref(self.context), C.byref(self.error)))
        self.addCleanup(
            lambda: self.ok(self.api.destroy(self.context, C.byref(self.error)))
        )

    def ok(self, status):
        self.assertEqual(status, 0, self.error.message.decode())

    def request(self, bits):
        return {
            "kernel": "test",
            "target": {"arch": self.target.arch, "warp_size": self.target.warp_size},
            "grid": [8, 1, 1],
            "signature": {
                "src": f"*u{bits}",
                "dst": f"*u{bits}",
                "mask": "*u8",
                **{
                    name: "i32"
                    for name in (
                        "width",
                        "height",
                        "channels",
                        "src_stride",
                        "dst_stride",
                        "mask_stride",
                    )
                },
            },
            "constants": {"BLOCK": 128},
        }

    def test_compile_and_launch(self):
        torch = self.torch
        self.register()
        identity = C.create_string_buffer(256)
        self.ok(
            self.api.device_id(
                self.context, identity, len(identity), C.byref(self.error)
            )
        )
        self.assertIn(self.target.arch.encode(), identity.value)
        self.assertNotEqual(
            self.api.device_id(self.context, identity, 1, C.byref(self.error)), 0
        )
        bad_context = P(123)
        self.assertNotEqual(
            self.api.create(-1, C.byref(bad_context), C.byref(self.error)), 0
        )
        self.assertIsNone(bad_context.value)
        # Odd extents, multiple channels, unequal pitches and masked destination.
        width, height, channels = 19, 7, 3
        for bits, dtype in ((8, torch.uint8), (16, torch.uint16)):
            response = PROVIDER.compile(self.request(bits), root=self.root)
            self.assertEqual(response["format"], "hsaco")
            self.assertEqual(response["block"], [4 * self.target.warp_size, 1, 1])
            artifact = self.root / f"test{bits}.hsaco"
            artifact.write_bytes(base64.b64decode(response["artifact"]))
            kernel = P()
            self.assertNotEqual(
                self.api.load(
                    self.context,
                    str(artifact).encode(),
                    b"missing_entry",
                    C.byref(kernel),
                    C.byref(self.error),
                ),
                0,
            )
            self.assertIsNone(kernel.value)
            self.ok(
                self.api.load(
                    self.context,
                    str(artifact).encode(),
                    response["entry"].encode(),
                    C.byref(kernel),
                    C.byref(self.error),
                )
            )
            try:
                for use_default in (False, True):
                    stream = (
                        torch.cuda.default_stream()
                        if use_default
                        else torch.cuda.Stream()
                    )
                    with torch.cuda.stream(stream):
                        src = torch.arange(height * 64, dtype=torch.int32).reshape(
                            height, 64
                        )
                        src = src.to(device="cuda", dtype=dtype)
                        dst = torch.full((height, 68), 0x33, device="cuda", dtype=dtype)
                        mask = torch.arange(height * 23).reshape(height, 23) % 3 != 0
                        mask = mask.to(device="cuda", dtype=torch.uint8)
                        values = [
                            P(src.data_ptr()),
                            P(dst.data_ptr()),
                            P(mask.data_ptr()),
                        ]
                        values += [
                            C.c_int32(n) for n in (width, height, channels, 64, 68, 23)
                        ]
                        values += [P() for _ in response["implicit_null_arguments"]]
                        arguments = (P * len(values))(
                            *[C.cast(C.pointer(v), P) for v in values]
                        )
                        config = Config(
                            (C.c_uint32 * 3)(*response["grid"]),
                            (C.c_uint32 * 3)(*response["block"]),
                            response["shared_memory_bytes"],
                        )
                        native_stream = None if use_default else P(stream.cuda_stream)
                        self.ok(
                            self.api.launch(
                                self.context,
                                kernel,
                                C.byref(config),
                                arguments,
                                native_stream,
                                C.byref(self.error),
                            )
                        )
                        # Repeated launch reuses the same loaded module.
                        self.ok(
                            self.api.launch(
                                self.context,
                                kernel,
                                C.byref(config),
                                arguments,
                                native_stream,
                                C.byref(self.error),
                            )
                        )
                    stream.synchronize()  # Only the caller synchronizes.
                    actual = dst.cpu().to(torch.int32)
                    expected = torch.full((height, 68), 0x33, dtype=torch.int32)
                    enabled = (
                        mask.cpu()[:, :width].repeat_interleave(channels, dim=1).bool()
                    )
                    pixels = src.cpu().to(torch.int32)[:, : width * channels] ^ 0x5A
                    expected[:, : width * channels] = torch.where(enabled, pixels, 0x33)
                    self.assertTrue(
                        torch.equal(actual, expected), f"{bits}-bit mismatch"
                    )
                config.grid[0] = 0
                self.assertNotEqual(
                    self.api.launch(
                        self.context,
                        kernel,
                        C.byref(config),
                        arguments,
                        None,
                        C.byref(self.error),
                    ),
                    0,
                )
            finally:
                self.ok(self.api.unload(self.context, kernel, C.byref(self.error)))

    def test_invalid_compile_requests(self):
        self.register()
        for field, value in (
            ("grid", [0, 1, 1]),
            ("grid", [True, 1, 1]),
            ("target", {"arch": "sm_80", "warp_size": 32}),
            ("target", {"arch": "gfx1201", "warp_size": 16}),
            ("options", {"num_ctas": 2}),
            ("options", {"num_warps": 3}),
            ("options", {"arch": "gfx942"}),
            ("options", {"launch_cooperative_grid": True}),
            ("signature", {"not_an_argument": "i32"}),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                PROVIDER.compile({**self.request(8), field: value}, root=self.root)

    def test_compiler_process(self):
        self.register()
        # Mirror the installed share/flagcv layout, outside the source tree.
        shutil.copytree(
            ROOT / "compiler/flagcv_codegen",
            self.root / "compiler/flagcv_codegen",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy(
            ROOT / "backends/amd/compiler.py", self.root / "backends/amd/compiler.py"
        )
        command = [sys.executable, "-m", "flagcv_codegen.main", "--backend", "amd"]
        env = {**os.environ, "PYTHONPATH": str(self.root / "compiler")}
        identified = subprocess.run(
            command + ["--identify"],
            cwd=self.root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        compiled = subprocess.run(
            command,
            cwd=self.root,
            env=env,
            input=json.dumps(self.request(8)),
            text=True,
            capture_output=True,
            check=True,
        )
        response = json.loads(compiled.stdout)
        self.assertEqual(
            response["compiler_id"], json.loads(identified.stdout)["compiler_id"]
        )
        self.assertTrue(base64.b64decode(response["artifact"]).startswith(b"\x7fELF"))
        self.assertEqual(
            response["implicit_null_arguments"], ["global_scratch", "profile_scratch"]
        )


if __name__ == "__main__":
    unittest.main()
