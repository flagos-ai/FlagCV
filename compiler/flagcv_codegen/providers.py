# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Launch plans shared by the platform providers.

A provider owns the platform-specific decisions. Most of them are the same
shape for every vendor — pick a tile, a warp count and a grid — and differ
only in the limits the platform imposes, which live in platform.py. Keeping
the shape here means a new operator is added once, not once per vendor.
"""

from flagcv_codegen.platform import platform

# Element sizes the kernels can be instantiated at.
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


def _power_of_two_at_most(value, floor=64):
    block = floor
    while block * 2 <= value:
        block *= 2
    return block


def plan_flip(request, facts=None):
    """Tile for flip: BLOCK_X channel values per program along a row.

    Each program moves BLOCK_Y x BLOCK_X values. Block sizes follow the
    vendor's element-block ceiling from FlagGems (`facts["tile"]`) divided by
    the element size, so wide elements do not blow past it, and are capped at
    a power of two for the vectorised axis.
    """
    facts = facts or platform(request["backend"])
    size = DEPTH_SIZE[request["images"][0]["depth"]]
    block_x = _power_of_two_at_most(max(facts["tile"] // size, 64))
    block_y = 4
    if facts["warp"] == 1:  # Ascend has no SIMT warps: one block per program row
        block_y = 1
    grid = [
        {"product": ["width", "CHANNELS"], "block": block_x},
        {"product": ["height"], "block": block_y},
        1,
    ]
    num_warps = min(4, facts["max_warps"])
    return {"BLOCK_X": block_x, "BLOCK_Y": block_y}, num_warps, grid


PLANS = {"flip": plan_flip}


def compile_with_plan(request):
    """Dispatch a compile request to the kernel's plan for this platform."""
    facts = platform(request["backend"])
    plan = PLANS.get(request["kernel"])
    if plan is None:
        raise ValueError(
            f"{request['backend']} has no launch plan for kernel {request['kernel']!r}"
        )
    tuning, num_warps, grid = plan(request, facts)
    return tuning, num_warps, grid, facts
