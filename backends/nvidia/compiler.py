# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA compiler provider.

Two responsibilities:

* ``identify()`` returns a digest of everything that can change generated
  code. The runtime folds it into the artifact cache key, so a toolchain or
  kernel source change invalidates cached artifacts automatically.
* ``compile()`` resolves the kernel through the platform-then-common registry
  and compiles the selected variants.

Kernel capability decisions live here rather than in lowering: block sizes,
shared memory limits and maximum grid extents are platform properties, so
lowering stays unchanged when a new platform is added.
"""

PROVIDER_NAME = "nvidia_triton"


def identify():
    """TODO: return a digest over the toolchain and kernel sources.

    Must cover at least: the Python interpreter, the Triton distribution and
    its version, this file, and every kernel source reachable through both
    registries.
    """
    raise NotImplementedError


def compile(request):
    """TODO: compile the kernel named by `request`.

    Steps:
      1. Resolve the operation against backends/nvidia/kernels/registry.json,
         falling back to kernels/registry.json when unregistered here.
      2. Reject the request when a platform-registered kernel's constraints do
         not hold, rather than silently switching to the common kernel --
         platform registration is exclusive, and dispatch/ has already checked
         those constraints at build time.
      3. Expand autotune candidates from the registry's tuning section.
      4. Compile each variant and report grid, warp count and signature.
    """
    raise NotImplementedError
