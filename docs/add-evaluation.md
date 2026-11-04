# Evaluate the add kernel

The Python evaluation uses the same first profile as
[ops-baseline's OpenCV CUDA add](https://github.com/zhaojs12345/ops-baseline/tree/3b2f022314b0d6e3bf6857f400c31beb65a505dd/ops/opencv).
It covers four image sizes with float32 and uint8, contiguous single-channel
inputs, no mask and an independent preallocated destination. A candidate
implements `add(a, b, dst)`, writes all of `dst`, returns that same tensor and
leaves both inputs unchanged. The uint8 result saturates at 255.

The Definition describes this Python callable. It is independent of the
C++ `Handle` and `GpuMat` integration, which has its own CTest entry points.

Run from the repository root. Case listing only requires pytest. Executing
cases requires Torch and the requested device runtime. Timing uses the
backend's Triton benchmark helper.

```sh
python -m pytest benchmark/test_add_benchmark.py -m add --level core --list-cases --output add-cases.json
python -m pytest tests/test_add.py -m add --device cpu --ref-as-candidate
python -m pytest tests/test_add.py benchmark/test_add_benchmark.py -m add --device cpu --dummy-candidate add
python -m pytest tests/test_add.py -m add --override add:/path/to/candidate.py
python -m pytest benchmark/test_add_benchmark.py -m add --override add:/path/to/candidate.py \
  --case-id 'benchmark/test_add_benchmark.py::test_add_benchmark::core::uint8::3' --base-data /path/to/op_opencv_perf_baseline.json \
  --output add-result.json
```

Append `:FUNCTION` to the override specification if the candidate entry point
has another name. The default device is CUDA. Use `--reference FILE:FUNCTION`
to additionally check an independent oracle with the same signature. This
compares the Torch reference and candidate against the supplied oracle.

Benchmark results compare device microseconds with the supplied
OpenCV CUDA record's `T_us`. Lookup requires an exact shape, dtype and
OpenCV profile match. The reported speedup is the raw latency ratio.
Dummy runs report candidate reachability, and reference-as-candidate runs
exercise the test harness. Use an explicit candidate for performance claims.
