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

## Pinned FlagTree / ROCm 7.2.4 environment

FlagTree is FlagOS's Triton compiler and is imported as `triton`. The
[official AMD installation guide](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-amd)
publishes `flagtree==0.7.0+amd3.6` for Python 3.12. The project Dockerfile
combines that wheel with ROCm 7.2.4 and the base image's PyTorch 2.10, without
replacing their shared runtime libraries:

```sh
docker build -t flagcv:flagtree-rocm7.2.4 -f docker/amd/Dockerfile docker/amd
docker run --rm --device /dev/kfd --device /dev/dri --shm-size 512m \
  -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 \
  -e TRITON_CACHE_DIR=/workspace/build/cache/flagtree-rocm724/triton \
  -e TORCH_EXTENSIONS_DIR=/workspace/build/cache/flagtree-rocm724/torch \
  -v "$PWD:/workspace" flagcv:flagtree-rocm7.2.4 bash -c '
    set -e
    cmake -S . -B build/flagtree-rocm724 \
      -DFLAGCV_BACKENDS=amd -DFLAGCV_WARNINGS_AS_ERRORS=ON \
      -DPython3_EXECUTABLE=/opt/flagcv-env/bin/python \
      -DFLAGCV_TEST_AMD_GPU=ON -DFLAGCV_TEST_AMD_AOT_TARGET=gfx1201:32 \
      -DCMAKE_INSTALL_PREFIX=/workspace/build/install-flagtree-rocm724
    cmake --build build/flagtree-rocm724 -j4
    ctest --test-dir build/flagtree-rocm724 --output-on-failure -j1
  '
```

This selects physical GPU0 and exposes it as ordinal 0. Normal HIP processes
can share it with existing workloads; no exclusive reservation or special
attach service is needed. The tests allocate small fixtures, use caller-owned
streams, and wait only for their own work. Shared-card results establish
correctness, not isolated performance.

The base image digest, official wheel SHA256, compiler version and package
identities are recorded in
[`flagtree-rocm7.2.4.lock.json`](../docker/amd/flagtree-rocm7.2.4.lock.json).
FlagTree is installed in `/opt/flagcv-env`; the base Torch dependencies remain
available through a `.pth` file. `--identify` reports the distribution owning
the actual imported module, so an inherited stock Triton installation is not
mistaken for the active FlagTree compiler. Keep native builds and caches in
separate directories for each Python/compiler/ROCm profile.

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

Use a compatible FlagTree/Triton installation with the AMD backend and
`ASTSource`/`GPUTarget` AOT APIs. Validated compiler profiles are FlagTree
0.7.0+amd3.6 (Triton 3.6) and AMD Triton 3.8. Compilation uses an explicit
target and does not initialize a GPU or import PyTorch.

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
`argument_names` and `signature`, `target`, `compiler`, `compiler_id`, and
`implicit_null_arguments`. Decode the artifact to a file before loading it.
Append one host-stored null device pointer for each `implicit_null_arguments`
entry, in the returned order, after the explicit arguments. Triton appends
global and profile scratch pointers even when scratch sizes are zero.
FlagTree/Triton 3.6 uses a permanently-null global scratch placeholder without
publishing its size. Triton 3.8 publishes both sizes. The provider handles
these two metadata conventions, rejects nonzero scratch requirements, and
rejects unknown conventions with missing size metadata.

`identify()` hashes the interpreter identity, the active FlagTree/Triton
distribution identity and compiler/backend code (including bundled bitcode and
linker binaries), this provider, compiler entry sources, both registries,
and Python sources in both kernel trees. A future runtime cache must also key
on target/device, signature, constants, grid and options. Pixel values and
pointer addresses must never enter that key. Diagnostics go to stderr and a
failed compiler process exits nonzero.

## Validation

CPU registry and compiler-identity tests run with AMD selected. Real AOT tests
can run without GPU access by setting `FLAGCV_TEST_AMD_AOT_TARGET=gfx1201:32`;
CTest explicitly hides GPUs from that test process. Hardware tests are opt-in
and additionally need compatible PyTorch and FlagTree/Triton installations:

```sh
cmake -S . -B build/amd \
  -DFLAGCV_BACKENDS=amd -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DFLAGCV_TEST_AMD_AOT_TARGET=gfx1201:32 \
  -DFLAGCV_WARNINGS_AS_ERRORS=ON -DFLAGCV_TEST_AMD_GPU=ON
cmake --build build/amd -j
HIP_VISIBLE_DEVICES=0 ctest --test-dir build/amd --output-on-failure
```

The following profiles pass all four CTest groups on AMD Radeon AI PRO R9700
(`gfx1201`, wave32). The November 2 checks used one shared physical GPU0:

| Profile | HIP | Python | PyTorch distribution | Active compiler |
| --- | --- | --- | --- | --- |
| ROCm 7.2.4 | 7.2.53211 | 3.12.3 | 2.10.0+rocm7.2.4.lw.git3d3aa833 | FlagTree 0.7.0+amd3.6 / Triton 3.6.0 |
| ROCm 10.1.0 | 7.16.26385 | 3.13.13 | 2.14.0+rocm10.1.0 | AMD Triton 3.8.0+git669b31ac.rocm10.1.0 |

Coverage includes:

- Build with warnings treated as errors; plugin discovery, Handle moves,
  lifecycle, invalid backend/device errors and HIP-backed GpuMat strides.
- Common/AMD registry precedence, exclusive failures, duplicate/missing
  entries, traversal rejection and source/registry cache invalidation.
- External compiler subprocesses using the installed directory layout.
- Actual Triton HSACO loading and repeated HIP launches on default and
  non-default streams: exact 8/16-bit masked results on odd image extents,
  three channels, unequal pitches, and preservation of masked pixels/padding.
- Missing symbols and invalid compiler/launch requests.
- Recovery from a handled HIP error before subsequent caller-owned PyTorch
  operations, and accurate FlagTree identity when stock Triton metadata is
  also visible.

The ROCm 10.1 qualification image is
`sha256:a79c374ab2bc93c0e4358da6c7bda5cb84cf1308e8aff74fc99fc95c511cbd03`.
Other GPU architectures have not been hardware-qualified by these checks.

The repository still has no public operators. The AMD test kernel lives only
under `tests/fixtures`; no test operator is installed or registered. The common
runtime's `launch()`, lowered-operation request construction and compiled-kernel
cache remain unimplemented. These tests qualify the AMD compiler/plugin path,
not public image-operator execution or performance.
