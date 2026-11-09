# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon (DCU) compiler provider.

Kernel capability decisions live here, not in lowering: block sizes, shared
memory limits and grid extents are platform properties, so lowering stays
unchanged when a platform is added. The plan itself is shared
(flagcv_codegen.providers) and bounded by this platform's limits in
flagcv_codegen.platform; what is specific here is the identity digest and any
kernel this platform registers itself, which would be resolved through
backends/hygon/kernels/registry.json first and never fall back to the common
kernel.
"""

from flagcv_codegen.providers import compile_with_plan
from flagcv_codegen.triton_build import compile_kernel, digest

PROVIDER = "hygon"


def identify():
    """Digest of the interpreter, Triton, this file and every registered kernel."""
    return digest(__file__, PROVIDER)


def compile(request):
    tuning, num_warps, grid, _ = compile_with_plan(request)
    return compile_kernel(request, tuning, num_warps, grid)
