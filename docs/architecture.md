# FlagCV Architecture

FlagCV is an image processing library with OpenCV-shaped interfaces and
FlagDNN-shaped multi-backend plumbing. It runs in **immediate mode** only: a
call does its work and returns. There is no graph to build, no compile step the
caller drives, and no cross-operator fusion.

## Layers

```
include/flagcv/        public C++ API (OpenCV-shaped)
      |
src/core, src/imgproc  one source file per operator family
      |
src/lowering           operands + params -> LoweredOp (validation lives here)
      |
src/runtime            cache lookup, compile on miss, launch
      |
compiler/              out-of-process Triton compiler (Python)
      |
backends/<platform>/   dlopen'd plugin: context, module load, launch
```

Each arrow is one-way. The core library links no vendor SDK and no Python; it
reaches the toolchain only by talking to the external compiler process, and
reaches the device only through a backend plugin.

## Kernel resolution

Two registries are consulted, in order:

1. `backends/<platform>/kernels/registry.json`
2. `kernels/registry.json` (common Triton kernels)

Platform registration is **exclusive**. Once a platform claims an operator,
a failure there is reported to the caller — the common kernel is never silently
tried as a fallback. So every constraint a platform kernel imposes (depth,
channel count, alignment) must be checked in that platform's `compiler.py`
before compilation, not discovered during it.

## Compile and launch path

```
flagcv::flip(handle, src, dst, code)
  lower_flip()                       validate, build LoweredOp
  launch()                           cache key -> Handle's kernel cache
    miss: resolve_kernel()           platform registry, then common
          CompilerClient::request()  JSON line to the compiler process
            python3 -m flagcv_codegen.main     (started once, kept alive)
              backends/<platform>/compiler.py  choose blocks/warps/grid
              triton.compile(target from the plugin)
              -> artifact file + launch metadata
          plugin.load(artifact bytes)
    plugin.launch(grid, args, stream)  no compiling, no synchronizing
```

The compiler process is started on the first cache miss and serves every
`Handle` in the host process, so the Python and Triton import is paid once
per process rather than per compile. It never initialises a device: the
plugin reports the compile target (`cuda:90:32`), so the compiler imports no
framework and needs no GPU. The protocol is one JSON object per line; it is
documented in `compiler/flagcv_codegen/main.py`.

**Cache key.** Kernel name, each image's depth, channel count and whether its
base pointer is 16-byte aligned, compile-time param values, runtime param
types, the device, and the compiler identity (a digest of the interpreter,
Triton version, provider and every registered kernel source). Image sizes and
strides are runtime params, so a new size reuses the kernel. A runtime param
the lowering marks with `add_hinted_param` is also specialised on
divisibility by 16, as Triton's own JIT does, so row accesses can vectorise.

**Files.** Compiled binaries go to `$FLAGCV_CACHE_DIR/artifacts/<hash>/`
(default `~/.cache/flagcv`); Triton's own cache (`~/.triton/cache`) holds the
intermediate IR. The runtime's kernel cache lives in memory per `Handle`, so
a new host process restarts the compiler process; the Triton cache makes its
compiles cheap, but the process start (~2 s including Triton import) is paid
again.

**Locations and overrides.** `FLAGCV_PYTHON` selects the interpreter for the
compiler process (default `python3`). Kernels, the compiler and platform
providers are found under `FLAGCV_DATA_DIR`, else `../share/flagcv` next to
libflagcv, else the source tree it was built from. Plugins are found under
`FLAGCV_PLUGIN_DIR`, else `flagcv/` next to libflagcv, else the build tree.

## Units: pitch vs stride

The public API takes `row_pitch` in **bytes**, because imaging hardware pads
rows for DMA and that padding is independent of `width * channels *
depth_size`. Kernel pointers are typed by pixel depth, so their arithmetic
advances in **elements**.

`GpuMat::row_stride()` is the single place that conversion happens. Everything
below the public API — `ImageSpec::row_stride`, the params a `LoweredOp`
carries, kernel arguments — is in elements.

## Masks

A mask means "do not write", not "write zero". Destination pixels the mask
excludes keep their prior contents byte-identical. Kernels implement this by
folding the mask into the store predicate, never with
`tl.where(mask, result, 0)`. A masked call therefore requires an initialized
`dst`.

## Memory ownership

FlagCV never allocates or frees device memory. `GpuMat` is a non-owning view
over a buffer the caller owns — unlike `cv::cuda::GpuMat`, which owns its
data. Lifetime is the caller's problem.

## Plugin boundary

Plugins include `backends/backend.hpp` and nothing else from the tree. They
receive operand layouts and params as plain data, so adding a field to a core
struct is not an ABI break. A plugin that includes a core internal header has
broken the contract.
