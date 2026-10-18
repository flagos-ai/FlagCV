# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar compiler provider.

TODO: implement identify() and compile().

This platform registers no bitwise kernel, so those operations resolve to the
common registry. Kernel capability decisions (block sizes, shared memory,
maximum grid extents) belong here rather than in lowering.
"""

PROVIDER_NAME = "iluvatar_triton"


def identify():
    raise NotImplementedError


def compile(request):
    raise NotImplementedError
