# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA compiler provider.

Kernel capability decisions live here, not in lowering: block sizes, shared
memory limits and grid extents are platform properties, so lowering stays
unchanged when a platform is added.

Resolution order is this platform's kernels/registry.json, then the common
kernels/registry.json; the runtime resolves it and names the chosen source in
the request. Platform registration is EXCLUSIVE: once an operation is
registered here, a constraint failure must be reported, not silently retried
against the common kernel. So every layout constraint a platform kernel relies
on gets checked here, before compiling, where the operand layout is fully
known.
"""

from flagcv_codegen.triton_build import DEPTH_SIZE, compile_kernel, digest

PROVIDER = "nvidia"


def identify():
    """Digest of everything that can change generated code.

    The Python interpreter, the Triton version, this file, the shared codegen
    helpers and every kernel source reachable through both registries. The
    runtime folds it into the cache key, so a kernel edit invalidates cached
    artifacts.
    """
    return digest(__file__, PROVIDER)


def _plan_flip(request):
    # Each program moves a BLOCK_Y x BLOCK_X tile of channel values. Narrow
    # depths get wider rows so every program still moves ~2-4 KiB.
    size = DEPTH_SIZE[request["images"][0]["depth"]]
    block_x = {1: 1024, 2: 512, 4: 256, 8: 128}[size]
    block_y = 4
    grid = [
        {"product": ["width", "CHANNELS"], "block": block_x},
        {"product": ["height"], "block": block_y},
        1,
    ]
    return {"BLOCK_X": block_x, "BLOCK_Y": block_y}, 4, grid


_PLANS = {"flip": _plan_flip}


def compile(request):
    plan = _PLANS.get(request["kernel"])
    if plan is None:
        raise ValueError(
            f"nvidia provider has no launch plan for kernel {request['kernel']!r}"
        )
    tuning, num_warps, grid = plan(request)
    return compile_kernel(request, tuning, num_warps, grid)
