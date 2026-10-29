# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""External compiler entry point.

The core runtime invokes this as a separate process on a cache miss, passing
the operation and its operand layouts. Keeping it out-of-process is what lets
the core library link no Python and no platform toolchain.

Read one JSON request from stdin and return one JSON response on stdout.
Diagnostics go to stderr. Providers define their own compilation contract.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--identify", action="store_true")
    args = parser.parse_args()
    try:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", args.backend):
            raise ValueError("invalid backend name")
        root = Path(__file__).resolve().parents[2]
        source = root / "backends" / args.backend / "compiler.py"
        spec = importlib.util.spec_from_file_location("flagcv_provider", source)
        provider = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(provider)
        response = (
            {"compiler_id": provider.identify()}
            if args.identify
            else provider.compile(json.load(sys.stdin))
        )
        json.dump(response, sys.stdout)
        sys.stdout.write("\n")
    except Exception as error:
        print(f"flagcv_codegen: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
