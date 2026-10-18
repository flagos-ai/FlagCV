# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon compiler provider.

TODO: implement identify() and compile().

Kernel capability decisions live here, not in lowering: block sizes, shared
memory limits and grid extents are platform properties, so lowering stays
unchanged when a platform is added.

This also decides which kernel to use. Resolution order is this platform's
kernels/registry.json, then the common kernels/registry.json. Platform
registration is EXCLUSIVE: once an operation is registered here, a constraint
failure must be reported, not silently retried against the common kernel. So
every layout constraint a platform kernel relies on gets checked here, before
compiling, where the operand layout is fully known.
"""

PROVIDER = "hygon"


def identify():
    """TODO: digest everything that can change generated code.

    At minimum the Python interpreter, the Triton version, this file and every
    kernel source reachable through both registries. The runtime folds it into
    the cache key, so a kernel edit invalidates cached artifacts.
    """
    raise NotImplementedError


def compile(request):
    """TODO: compile the kernel named by `request`."""
    raise NotImplementedError
