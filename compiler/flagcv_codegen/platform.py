# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-platform facts the compiler needs, keyed by backend.

Everything here is observed from the platform or from libtriton_jit's backend
headers for the same platform, not guessed per vendor: the Triton backend
name, the artifact format Triton emits, and the tuning shape FlagGems uses for
element-copy work on that vendor.

`binary` is the key in `compiled.asm` holding the image the plugin loads;
`extension` is what the artifact file is named. `arch` is consulted only when
the plugin did not report a target (the backend providers pass the plugin's
own string through). `tile` and `max_warps` bound the block a provider picks:
`tile` is the largest element block FlagGems uses for that vendor and
`max_warps` its per-CTA warp ceiling.
"""

PLATFORM = {
    "nvidia": {
        "triton_backend": "cuda",
        "binary": "cubin",
        "extension": "cubin",
        "warp": 32,
        # FlagGems caps NVIDIA tiles at 512 elements; this table uses 1024
        # because the flip measurements in this repo (docs/architecture.md)
        # selected 1024/512/256/128 channel values for 1/2/4/8-byte depths,
        # which is exactly 1024 // element size.
        "tile": 1024,
        "max_warps": 32,
    },
    "iluvatar": {
        "triton_backend": "ix",
        "binary": "cubin",
        "extension": "cubin",
        "warp": 64,
        # FlagGems: ILUVATAR max_tile_size 1024, max warps 32.
        "tile": 1024,
        "max_warps": 32,
    },
    "ascend": {
        "triton_backend": "npu",
        "binary": "npubin",
        "extension": "npubin",
        "warp": 1,
        # Ascend launches a fixed 48 AICore grid; FlagGems caps tiles at 512.
        "tile": 512,
        "max_warps": 32,
        "grid_limit": (48, 1, 1),
        # Ascend binaries are registered with the runtime, not loaded as a module.
        "load_flags": 1,
    },
    "hygon": {
        "triton_backend": "hcu",
        "binary": "hsaco",
        "extension": "hsaco",
        "warp": 64,
        # FlagGems: HYGON max_tile_size 2048, max warps 16.
        "tile": 2048,
        "max_warps": 16,
    },
    "mthreads": {
        # Triton's MThreads backend is named "mtgpu" (libtriton_jit maps it too).
        "triton_backend": "mtgpu",
        "binary": "mubin",
        "extension": "mubin",
        "warp": 32,
        # FlagGems: MTHREADS max_tile_size 1024; MUSA caps a block at 4 warps.
        "tile": 1024,
        "max_warps": 4,
    },
    "thead": {
        "triton_backend": "PPU",
        "binary": "hgbin",
        "extension": "hgbin",
        "warp": 32,
        # T-Head's FlagGems heuristics note larger blocks for memory
        # bandwidth; start from 1024 and keep the warp count conservative.
        "tile": 1024,
        "max_warps": 8,
    },
}


def platform(backend):
    entry = PLATFORM.get(backend)
    if entry is None:
        raise ValueError(f"no platform facts for backend {backend!r}")
    return entry
