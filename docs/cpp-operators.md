# C++ operator profiles

The C++ API is declared in `include/flagcv/core.hpp` and exported through
`<flagcv/flagcv.hpp>`. These profiles cover the array-array operations in the
Python evaluation suite.

| Operator | Depths | Channels | Behaviour |
| --- | --- | --- | --- |
| `add` | 8U, 32F | 1–4 | 8U saturates to [0, 255] |
| `subtract` | 8U, 32F | 1–4 | 8U saturates to [0, 255] |
| `multiply` | 8U, 32F | 1–4 | Unit scale; 8U saturates to [0, 255] |
| `bitwise_and` | 8U | 1–4 | Bitwise AND |
| `bitwise_not` | 8U | 1–4 | Bitwise complement |
| `flip` | All declared depths | 1–4 | Bit-exact; zero reverses rows, positive reverses columns, negative reverses both |

Inputs and destination must have equal extents, depth and channel count.
Each operand may have an independent byte pitch and a base offset aligned to
its element size. Pointwise operations accept exact source/destination aliasing
with equal pitches and reject partial overlap; `flip` rejects all overlap.
Scalar, mask, output-depth conversion and non-unit multiply scale overloads
are outside these profiles. Unsupported layouts and depths fail in lowering.

The caller allocates device buffers and owns their lifetime. Calls enqueue on
the supplied native stream (default `nullptr`); the caller synchronizes before
reading or releasing buffers. For example, with allocated `src1`, `src2`, `dst`:

```cpp
#include <flagcv/flagcv.hpp>

flagcv::Handle handle("nvidia", 0);
flagcv::add(handle, src1, src2, dst, stream);
flagcv::subtract(handle, src1, src2, dst, stream);
flagcv::multiply(handle, src1, src2, dst, stream);
flagcv::bitwise_and(handle, src1, src2, dst, stream);  // 8U operands
flagcv::bitwise_not(handle, src1, dst, stream);       // 8U operands
flagcv::flip(handle, src1, dst, -1, stream);          // separate destination
```

The entry points call family lowering, then the shared runtime. The runtime
resolves `kernels/registry.json`, asks the external compiler process for a
binary on a cache miss, and launches through the backend plugin. Five
pointwise operations share a pitched Triton kernel with a compile-time mode;
`flip` has its own kernel. Sizes and strides remain runtime arguments, with
alignment/divisibility specializations, so compatible sizes reuse binaries.
All six provider plans register these operations; hardware validation here
covers NVIDIA only.

## Validation

```bash
cmake -S . -B build-cpp-operators -DBUILD_SHARED_LIBS=ON \
  -DFLAGCV_BUILD_TESTS=ON -DFLAGCV_WARNINGS_AS_ERRORS=ON
cmake --build build-cpp-operators -j 8
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build-cpp-operators --output-on-failure
CUDA_VISIBLE_DEVICES=0 python benchmark/compare_cpp_pointwise.py \
  --executable build-cpp-operators/tests/test_pointwise --output results.json
```

On NVIDIA H100 80GB HBM3, CUDA toolkit 12.9, PyTorch 2.11.0+cu130 and Triton
3.6.0, validation passed:

- C++ `flip`: 70 checks against the CPU reference.
- C++ pointwise: 610 checks, including CPU results, saturation boundaries,
  independent pitches, offset pointers, 1–4 channels, tiny/odd/1080p shapes,
  source and padding preservation, both source aliases, non-default streams,
  and invalid layout/depth/overlap rejection.
- Existing Python kernel accuracy suites: 113 tests.
- Python evaluation infrastructure: 46 tests.
- Build with warnings as errors, Black, isort, flake8 and `git diff --check`.
  The repository has no `.pre-commit-config.yaml`, so `pre-commit run --all-files`
  cannot run.

Performance samples below use contiguous single-channel 1080×1920 images.
Both paths use CUDA graphs and preallocated output. C++ captures 100 calls and
times 10 graph replays after warmup; Torch uses `do_bench_cudagraph`. Timings
exclude compilation and CPU submission overhead. The baseline is the existing
Torch reference, not OpenCV CUDA or the existing standalone Triton candidates.
The [full measurements](cpp-pointwise-performance.csv) cover 32 profiles from
64×64 through 2160×3840; these are informational measurements, not test thresholds.

| Operator | Dtype | C++ (µs) | Torch reference (µs) | Speedup |
| --- | --- | ---: | ---: | ---: |
| add | uint8 | 2.453 | 19.749 | 8.051× |
| add | float32 | 4.269 | 4.849 | 1.136× |
| subtract | uint8 | 2.646 | 19.228 | 7.267× |
| subtract | float32 | 4.355 | 9.707 | 2.229× |
| multiply | uint8 | 2.813 | 29.302 | 10.417× |
| multiply | float32 | 4.166 | 9.667 | 2.320× |
| bitwise_and | uint8 | 1.894 | 4.604 | 2.431× |
| bitwise_not | uint8 | 1.781 | 4.434 | 2.489× |
