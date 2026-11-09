# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Platform-neutral helpers for providers that compile through Triton.

A provider decides the platform-specific parts (tuning constexprs, warps,
grid, binary format) and hands them to compile_kernel(), which does the
signature construction, compilation and artifact write that every
Triton-based platform shares.
"""

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget

# Kernel pointer element types, keyed by the depth names the runtime sends.
DEPTH_TO_TRITON = {
    "8U": "u8",
    "8S": "i8",
    "16U": "u16",
    "16S": "i16",
    "32S": "i32",
    "16F": "fp16",
    "32F": "fp32",
    "64F": "fp64",
}

DEPTH_SIZE = {
    "8U": 1,
    "8S": 1,
    "16U": 2,
    "16S": 2,
    "32S": 4,
    "16F": 2,
    "32F": 4,
    "64F": 8,
}


def data_dir():
    return Path(os.environ["FLAGCV_DATA_DIR"]).resolve()


def cache_dir():
    override = os.environ.get("FLAGCV_CACHE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "flagcv"


def parse_target(target):
    """'cuda:90:32' -> GPUTarget. The runtime gets this from the plugin, so the
    compiler never initialises a device or imports a framework to learn it."""
    backend, arch, warp_size = target.split(":")
    return GPUTarget(backend, int(arch) if arch.isdigit() else arch, int(warp_size))


def registry_sources(backend):
    """Every kernel source reachable through the platform and common registries."""
    root = data_dir()
    sources = []
    for registry in (
        root / "backends" / backend / "kernels" / "registry.json",
        root / "kernels" / "registry.json",
    ):
        if not registry.is_file():
            continue
        sources.append(registry)
        for entry in json.loads(registry.read_text())["kernels"]:
            sources.append(registry.parent / entry["source"])
    return sources


def digest(provider_file, backend):
    """Hash everything that can change generated code for this provider."""
    h = hashlib.sha256()
    h.update(sys.version.encode())
    h.update(triton.__version__.encode())
    files = [Path(provider_file)]
    files += sorted(Path(__file__).parent.glob("*.py"))
    files += registry_sources(backend)
    for path in files:
        h.update(str(path).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def load_jit_function(source, function):
    path = Path(source).resolve()
    module_name = "flagcv_kernel_" + hashlib.sha1(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, function)
    # Autotuner/Heuristics wrap the JITFunction in .fn.
    while not isinstance(fn, triton.runtime.JITFunction):
        fn = fn.fn
    return fn


def _check_arguments(fn, request, tuning):
    """The kernel's argument list must be exactly: one pointer per image, the
    lowered params in order, then the provider's tuning constexprs. Checking
    names here turns lowering/kernel drift into a clear error instead of a
    mis-bound launch."""
    expected = [image["name"] + "_ptr" for image in request["images"]]
    expected += [param["name"] for param in request["params"]]
    names = list(fn.arg_names)
    if names[: len(expected)] != expected:
        raise ValueError(
            f"{request['function']} arguments {names[:len(expected)]} "
            f"do not match the lowered operands {expected}"
        )
    rest = names[len(expected) :]
    if set(rest) != set(tuning):
        raise ValueError(
            f"{request['function']} has trailing arguments {rest} after the lowered "
            f"operands, but the provider supplied tuning constexprs {sorted(tuning)}"
        )


def compile_kernel(request, tuning, num_warps, grid, num_stages=1, binary="cubin"):
    """Compile the requested kernel and write its binary into the FlagCV cache.

    `tuning` holds the provider's constexpr choices (block sizes); `grid` is the
    launch-grid spec returned to the runtime, which evaluates it against the
    runtime parameter values it never sent here.
    """
    fn = load_jit_function(request["source"], request["function"])
    _check_arguments(fn, request, tuning)

    signature, constexprs, attrs = {}, {}, {}
    for index, image in enumerate(request["images"]):
        signature[image["name"] + "_ptr"] = "*" + DEPTH_TO_TRITON[image["depth"]]
        if image.get("aligned16"):
            attrs[(index,)] = [["tt.divisibility", 16]]
    for offset, param in enumerate(request["params"]):
        if param["type"] == "constexpr":
            signature[param["name"]] = "constexpr"
            constexprs[param["name"]] = param["value"]
        elif param["type"] in ("i32", "i64"):
            signature[param["name"]] = param["type"]
            if param.get("div16"):
                attrs[(len(request["images"]) + offset,)] = [["tt.divisibility", 16]]
        else:
            raise ValueError(f"unsupported parameter type {param['type']!r}")
    for name, value in tuning.items():
        signature[name] = "constexpr"
        constexprs[name] = value

    target = parse_target(request["target"])
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    # Deliberately no fn.create_binder(): it probes the active driver, which
    # imports torch and initialises a device. ASTSource compilation does not
    # need it, and the explicit target keeps this process device-free.
    source = triton.compiler.ASTSource(
        fn=fn, signature=signature, constexprs=constexprs, attrs=attrs
    )
    compiled = triton.compile(source, target=target, options=options.__dict__)

    metadata = compiled.metadata
    # The runtime passes null scratch pointers; a kernel that needs real
    # scratch memory would silently corrupt memory, so refuse it here.
    for field in ("global_scratch_size", "profile_scratch_size"):
        if getattr(metadata, field, 0):
            raise RuntimeError(
                f"{request['function']} needs {field}={getattr(metadata, field)}, "
                "which is not supported"
            )

    artifact_dir = cache_dir() / "artifacts" / compiled.hash
    artifact = artifact_dir / f"{metadata.name}.{binary}"
    if not artifact.is_file():
        artifact_dir.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a concurrent process never loads a torn binary.
        with tempfile.NamedTemporaryFile(dir=artifact_dir, delete=False) as tmp:
            tmp.write(compiled.asm[binary])
        os.chmod(tmp.name, 0o644)
        os.replace(tmp.name, artifact)
    return {
        "artifact": str(artifact),
        "function": metadata.name,
        "num_warps": metadata.num_warps,
        "shared": metadata.shared,
        "grid": grid,
    }
