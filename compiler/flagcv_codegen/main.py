# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""External compiler entry point.

The core runtime invokes this as a separate process on a cache miss, passing
the operation and its operand layouts. Keeping it out-of-process is what lets
the core library link no Python and no platform toolchain.

TODO: implement.

  1. Read the request from stdin.
  2. Load the platform provider from backends/<platform>/compiler.py.
  3. Hand it the request; it resolves the kernel and compiles.
  4. Write back the artifact plus launch metadata (grid, warps, signature).
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
