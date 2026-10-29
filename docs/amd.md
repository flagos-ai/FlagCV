# AMD GPU backend

## Build and use

The AMD plugin uses the HIP host module API on Linux. Install a ROCm HIP SDK
with its CMake package, a C++17 compiler and CMake 3.23 or newer. No hipcc device
compilation or fixed GPU architecture is needed to build the plugin.

```sh
cmake -S . -B build/amd \
  -DFLAGCV_BACKENDS=amd \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/install"
cmake --build build/amd -j
cmake --install build/amd
```

`FLAGCV_BACKENDS` accepts a semicolon-separated list; selecting AMD does not
require a CUDA SDK. Leaving AMD out does not discover or link HIP.

```cpp
#include <flagcv/flagcv.hpp>

flagcv::Handle handle("amd", 0);  // Ordinal within HIP_VISIBLE_DEVICES.
// GpuMat accepts caller-owned hipMalloc memory; row_pitch is in bytes.
// A flagcv::Stream is the caller's hipStream_t cast to void*.
```

`Handle` loads `libflagcv_backend_amd.so` and validates its versioned entry
table. Plugin search order is the colon-separated `FLAGCV_BACKEND_PATH`, the
configured installation's `lib/flagcv/backends` directory (or `lib64` as selected
by CMake), then the dynamic loader's standard paths. For an uninstalled build:

```sh
export FLAGCV_BACKEND_PATH="$PWD/build/amd/backends/amd"
```

For a relocated installation, set this variable to the relocated plugin
directory. Missing plugins and invalid devices throw `flagcv::Error` during
`Handle` construction. The core has no HIP headers or HIP link dependency.

## Backend contract

The private `backends/backend.hpp` entry table is version 1. It binds a visible
device, reports a cache identity including PCI address, architecture, native
wave size and HIP runtime version, loads an HSACO path and entry symbol,
unloads modules, and launches kernels. Callbacks return a status and a bounded
error message; C++ exceptions and SDK types do not cross the plugin boundary.

Launch geometry is **blocks** in `grid[3]` and **threads** in `block[3]`.
`arguments` contains addresses of host argument storage, in compiled argument
order, including implicit arguments. A null stream selects the HIP default
stream. Launch does not compile, tune or synchronize. Backend calls restore
the host thread's previous HIP device.

The caller owns memory and streams. The plugin uses HIP's initialized device
state and never resets it. Synchronize outstanding work before unloading its
kernel or destroying its context; unload all kernels before destroying their
context. A `Handle` is confined to one host thread.

## External compiler

Use a compatible Triton/FlagTree installation with the AMD backend and
`ASTSource`/`GPUTarget` AOT APIs. The provider expects metadata for warp size,
global scratch and profile scratch; validation used Triton 3.8. Compilation
uses an explicit target and does not initialize a GPU or import PyTorch.

```sh
PYTHONPATH=compiler python -m flagcv_codegen.main --backend amd --identify
PYTHONPATH=compiler python -m flagcv_codegen.main --backend amd < request.json
```

In an installation, set `PYTHONPATH` to `<prefix>/share/flagcv/compiler`.
Compiler providers and registries are installed alongside it under
`<prefix>/share/flagcv/backends` and `<prefix>/share/flagcv/kernels`.

One JSON request describes a registered kernel's explicit specialization:

```json
{
  "kernel": "registered_kernel_name",
  "target": {"arch": "gfx1201", "warp_size": 32},
  "signature": {"src": "*u8", "dst": "*u8", "count": "i32"},
  "constants": {"BLOCK": 256},
  "grid": [4, 1, 1],
  "options": {"num_warps": 4}
}
```

This example requires a registry entry and a kernel accepting those arguments;
the production registries are intentionally empty. `signature` and `constants`
must partition all kernel arguments. Explicit signatures accept primitive
scalar and pointer types; tensor descriptors are not supported yet.
Grid counts are positive uint32 values.
Native wave sizes are 32 or 64; the compiled wave size must match the target.
Block geometry comes from compiler metadata, rather than a hardcoded CUDA warp
size. Cooperative launches, multiple CTAs and kernels requiring global/profile
scratch allocations are currently rejected.

The provider consults the AMD registry before the common registry. A claimed
AMD entry that fails validation or compilation is an error; it never falls
back to a common entry. Registry paths must remain inside their kernel tree.

The response contains base64-encoded `artifact` bytes in `hsaco` format,
the compiled `entry`, `grid`, `block`, `shared_memory_bytes`, explicit
`argument_names` and `signature`, `target`, `compiler_id`, and
`implicit_null_arguments`. Decode the artifact to a file before loading it.
Append one host-stored null device pointer for each `implicit_null_arguments`
entry, in the returned order, after the explicit arguments. Triton appends
global and profile scratch pointers even when scratch sizes are zero.

`identify()` hashes the interpreter identity, Triton distribution identity and
compiler/backend code, this provider, compiler entry sources, both registries,
and Python sources in both kernel trees. A future runtime cache must also key
on target/device, signature, constants, grid and options. Pixel values and
pointer addresses must never enter that key. Diagnostics go to stderr and a
failed compiler process exits nonzero.

## Validation

CPU registry and compiler-identity tests run with AMD selected. Hardware tests
are opt-in and additionally need compatible PyTorch and Triton installations:

```sh
cmake -S . -B build/amd \
  -DFLAGCV_BACKENDS=amd -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DFLAGCV_WARNINGS_AS_ERRORS=ON -DFLAGCV_TEST_AMD_GPU=ON
cmake --build build/amd -j
HIP_VISIBLE_DEVICES=0 ctest --test-dir build/amd --output-on-failure
```

Qualification on AMD Radeon AI PRO R9700 (`gfx1201`, wave32), ROCm 10.1.0,
HIP 7.16.26385, Python 3.13.13, PyTorch 2.14.0+rocm10.1.0 and
Triton 3.8.0+git669b31ac.rocm10.1.0 covers:

- Build with warnings treated as errors; plugin discovery, Handle moves,
  lifecycle, invalid backend/device errors and HIP-backed GpuMat strides.
- Common/AMD registry precedence, exclusive failures, duplicate/missing
  entries, traversal rejection and source/registry cache invalidation.
- External compiler subprocesses using the installed directory layout.
- Actual Triton HSACO loading and repeated HIP launches on default and
  non-default streams: exact 8/16-bit masked results on odd image extents,
  three channels, unequal pitches, and preservation of masked pixels/padding.
- Missing symbols and invalid compiler/launch requests.

The qualification image is
`sha256:a79c374ab2bc93c0e4358da6c7bda5cb84cf1308e8aff74fc99fc95c511cbd03`.
Other GPU architectures have not been hardware-qualified by these checks.

The repository still has no public operators. The AMD test kernel lives only
under `tests/fixtures`; no test operator is installed or registered. The common
runtime's `launch()`, lowered-operation request construction and compiled-kernel
cache remain unimplemented. These tests qualify the AMD compiler/plugin path,
not public image-operator execution or performance.
