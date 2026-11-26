# C++ operator interfaces

The C++ API is declared in `include/flagcv/core.hpp` and exported through
`<flagcv/flagcv.hpp>`. See the declarations for each operator's argument order,
layout requirements and numerical semantics.

## Operands and streams

`GpuMat` is a non-owning view of caller-allocated device memory. Inputs and
destination must satisfy the operator's extent, depth and channel requirements.
Each operand may have an independent byte pitch and a base offset aligned to
its element size. Lowering validates these requirements before compilation
or launch.

Pointwise operations accept exact source/destination aliasing with equal
pitches and reject partial overlap. Permutation operations may require
separate source and destination storage; check the API declaration before
using overlapping views.

Calls enqueue on the supplied native stream (default `nullptr`). The caller
owns the stream and device buffers and synchronizes before reading results
or releasing storage.

## Call path

The entry points call family lowering, then the shared runtime. The runtime
resolves `kernels/registry.json`, asks the external compiler process for a
binary on a cache miss, and launches through the backend plugin.

Pointwise operations share a pitched Triton kernel with a compile-time mode.
Sizes and strides remain runtime arguments, with alignment and divisibility
specializations, so compatible layouts reuse compiled binaries. Platform
providers choose the launch plan; backend plugins load the artifact and
enqueue execution on the caller's stream.

See [Architecture](architecture.md) for the compiler protocol, cache and plugin
boundary, and [Operator development](operator-development.md) for the steps to connect a
Triton operator to a C++ interface, lowering, kernel registry, compilation
plan and backend launch.
