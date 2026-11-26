# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""External compiler entry point.

The core runtime starts this once, as a long-lived child process, and sends it
every compile request for the life of the host process. Paying the Python and
Triton import once is what makes out-of-process compilation affordable;
keeping it out of process is what lets the core library link no Python and no
platform toolchain.

Protocol: one JSON object per line in each direction, over the child's stdin
and stdout. Every response echoes the request ``id`` and carries ``ok``; a
failure carries ``error`` instead of a result. Requests:

  {"id": 1, "op": "identify", "backend": "nvidia"}
      -> {"id": 1, "ok": true, "identity": "<hex digest>"}

  {"id": 2, "op": "compile", "backend": "nvidia", "target": "cuda:90:32",
   "kernel": "flip", "source": "/abs/kernels/common/flip.py",
   "function": "flip_kernel", "from_platform_registry": false,
   "images": [{"name": "src", "depth": "8U", "channels": 3, "aligned16": true}, ...],
   "params": [{"name": "width", "type": "i32"},
              {"name": "CHANNELS", "type": "constexpr", "value": 3}, ...]}
      -> {"id": 2, "ok": true, "artifact": "/abs/.../flip_kernel.cubin",
          "function": "flip_kernel", "num_warps": 4, "shared": 0,
          "grid": [{"product": ["width", "CHANNELS"], "block": 512},
                   {"product": ["height"], "block": 4}, 1]}

Runtime parameter values never reach this process: only their types do, so a
new image size reuses the compiled artifact. Each platform provider lives in
backends/<platform>/compiler.py and implements identify() and compile().
"""

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


def data_dir():
    return Path(os.environ["FLAGCV_DATA_DIR"]).resolve()


_PROVIDERS = {}


def load_provider(backend):
    provider = _PROVIDERS.get(backend)
    if provider is not None:
        return provider
    if not backend.replace("_", "").isalnum():
        raise ValueError(f"invalid backend name {backend!r}")
    path = data_dir() / "backends" / backend / "compiler.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"no compiler provider for backend {backend!r} at {path}"
        )
    spec = importlib.util.spec_from_file_location(f"flagcv_provider_{backend}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PROVIDERS[backend] = module
    return module


def handle(request):
    provider = load_provider(request["backend"])
    op = request.get("op")
    if op == "identify":
        return {"identity": provider.identify()}
    if op == "compile":
        return provider.compile(request)
    raise ValueError(f"unknown request op {op!r}")


def main():
    # Protocol replies get a private copy of stdout; fd 1 is then pointed at
    # stderr so stray prints from Triton or a provider cannot corrupt the
    # stream the runtime is parsing.
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)

    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            response = dict(handle(request), ok=True)
        # Every failure goes back to the caller rather than ending the process.
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        response["id"] = request_id
        replies.write(json.dumps(response) + "\n")
        replies.flush()


if __name__ == "__main__":
    main()
