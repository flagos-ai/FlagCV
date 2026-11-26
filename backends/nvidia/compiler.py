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

from flagcv_codegen.providers import compile_with_plan
from flagcv_codegen.triton_build import compile_kernel, digest

PROVIDER = "nvidia"


def identify():
    """Digest of everything that can change generated code.

    The Python interpreter, the Triton version, this file, the shared codegen
    helpers and every kernel source reachable through both registries. The
    runtime folds it into the cache key, so a kernel edit invalidates cached
    artifacts.
    """
    return digest(__file__, PROVIDER)


def compile(request):
    tuning, num_warps, grid, _ = compile_with_plan(request)
    return compile_kernel(request, tuning, num_warps, grid)
