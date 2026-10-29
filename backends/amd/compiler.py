# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""AMD Triton AOT provider. See docs/amd.md for the request/response contract."""

import base64
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys

PROVIDER = "amd"
ROOT = Path(__file__).resolve().parents[2]


def identify(*, root=ROOT):
    """Hash compiler/interpreter identity and both complete kernel source trees."""
    import triton

    digest = hashlib.sha256()
    try:
        version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        version = triton.__version__
    identity = (
        sys.version,
        str(Path(sys.executable).resolve()),
        version,
        str(Path(triton.__file__).resolve()),
    )
    digest.update(json.dumps(identity).encode())
    paths = {Path(__file__).resolve()}
    triton_root = Path(triton.__file__).resolve().parent
    for relative in ("compiler", "backends/amd", "_C"):
        directory = triton_root / relative
        if directory.is_dir():
            paths.update(directory.rglob("*.py"))
            paths.update(directory.rglob("*.so"))
    for directory in (root / "kernels", root / "backends/amd/kernels"):
        paths.update(directory.rglob("*.py"))
        paths.update(directory.rglob("*.json"))
    paths.update((root / "compiler/flagcv_codegen").glob("*.py"))
    for path in sorted(paths):
        digest.update(
            str(path.relative_to(root) if path.is_relative_to(root) else path).encode()
        )
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def resolve(kernel, *, root=ROOT):
    """A claimed platform entry is exclusive, including malformed entries."""
    for directory in (root / "backends/amd/kernels", root / "kernels"):
        registry = json.loads((directory / "registry.json").read_text())
        entries = [entry for entry in registry["kernels"] if entry["name"] == kernel]
        if not entries:
            continue
        if len(entries) != 1:
            raise ValueError(f"duplicate kernel registration: {kernel}")
        entry = entries[0]
        source = (directory / entry["source"]).resolve()
        if not source.is_relative_to(directory.resolve()) or not source.is_file():
            raise ValueError(f"invalid source for kernel {kernel}: {source}")
        return source, entry["function"]
    raise ValueError(f"no AMD or common kernel registered for {kernel!r}")


def compile(request, *, root=ROOT):
    """Compile an explicit signature to HSACO without initializing a GPU.

    Target and launch geometry are explicit, so cross-compilation is possible.
    No GPU driver, Torch or current-device detection is used by this provider.
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source, function = resolve(request["kernel"], root=root)
    target = request["target"]
    arch = target["arch"]
    wave = target["warp_size"]
    if not isinstance(arch, str) or not re.fullmatch(r"gfx[0-9a-f]+", arch):
        raise ValueError("target.arch must be an AMD gfx architecture")
    if type(wave) is not int or wave not in (32, 64):
        raise ValueError("target.warp_size must be 32 or 64")
    grid = request["grid"]
    if (
        not isinstance(grid, list)
        or len(grid) != 3
        or any(type(n) is not int or n <= 0 or n > 0xFFFFFFFF for n in grid)
    ):
        raise ValueError("grid must contain three positive uint32 block counts")
    options = dict(request.get("options", {}))
    warps = options.setdefault("num_warps", 4)
    if type(warps) is not int or warps not in (1, 2, 4, 8, 16) or warps * wave > 1024:
        raise ValueError("invalid num_warps for AMD target")
    if options.get("num_ctas", 1) != 1:
        raise ValueError("AMD module launches require num_ctas=1")
    if "arch" in options and options["arch"] != arch:
        raise ValueError("options.arch must match target.arch")
    if options.get("launch_cooperative_grid", False):
        raise ValueError("cooperative launches are not supported")
    signature = request["signature"]
    constants = request.get("constants", {})
    if not isinstance(signature, dict) or not isinstance(constants, dict):
        raise ValueError(
            "signature and constants must be objects keyed by argument name"
        )

    for value in signature.values():
        if not isinstance(value, str) or not re.fullmatch(
            r"\*?(?:[iu](?:1|8|16|32|64)|fp(?:16|32|64)|bf16)", value
        ):
            raise ValueError("only scalar and pointer signatures are supported")

    # The source module remains registered while Triton resolves JIT dependencies.
    module_name = "_flagcv_amd_" + hashlib.sha256(str(source).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        fn = getattr(module, function)
        args = set(fn.arg_names)
        if set(signature) | set(constants) != args or set(signature) & set(constants):
            raise ValueError(
                "signature and constants must partition the kernel arguments"
            )
        kernel = triton.compile(
            ASTSource(fn, signature=signature, constexprs=constants),
            target=GPUTarget("hip", arch, wave),
            options=options,
        )
        metadata = kernel.metadata
        if metadata.arch != arch:
            raise ValueError("compiled architecture does not match target")
        if metadata.warp_size != wave:
            raise ValueError("compiled warp size does not match target")
        # Triton appends these two pointers even when their buffers are unused.
        # Omitting them makes HIP read beyond the kernel argument array.
        if not all(
            hasattr(metadata, name)
            for name in ("global_scratch_size", "profile_scratch_size")
        ):
            raise ValueError("Triton must expose global/profile scratch metadata")
        if metadata.global_scratch_size or metadata.profile_scratch_size:
            raise ValueError("kernels requiring scratch buffers are not supported yet")
        return {
            "format": "hsaco",
            "artifact": base64.b64encode(kernel.asm["hsaco"]).decode("ascii"),
            "entry": metadata.name,
            "grid": grid,
            "block": [metadata.num_warps * wave, 1, 1],
            "shared_memory_bytes": metadata.shared,
            "argument_names": [name for name in fn.arg_names if name in signature],
            "signature": signature,
            "implicit_null_arguments": ["global_scratch", "profile_scratch"],
            "target": target,
            "compiler_id": identify(root=root),
        }
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
