# Python kernel evaluation

FlagCV's Python evaluation workflow supports case listing and selection,
candidate override, dummy candidate injection, reference as candidate, and
benchmark comparison with a live OpenCV CUDA baseline. Operator tests and benchmarks
use the root pytest options to expose these capabilities. Each operator defines
its own callable signature, inputs, and reference.

Python candidates run independently of the C++ integration.

Add a default reference in `reference/OP.py` with a callable named
`reference`, and mark its tests and benchmarks with `@pytest.mark.OP`. The
runner discovers operators from the reference modules and uses the marker to
select the reference or the matching `--override` entry. Dummy mode selects
only the operator named by `--dummy-candidate`.

## Example: add

Build the live baseline bridge using the
[OpenCV CUDA reference instructions](opencv-add-reference.md).
The add workload uses float32 and uint8, contiguous single-channel
inputs, no mask and an independent preallocated destination. A candidate
implements `add(a, b, dst)`, writes all of `dst`, returns that same tensor and
leaves both inputs unchanged. The uint8 result saturates at 255.

Run from the repository root. Case listing only requires pytest. Executing
cases requires Torch and the requested device runtime. Timing uses the
backend's Triton benchmark helper.

```sh
python -m pytest benchmark/test_add_benchmark.py -m add --level core --list-cases --output add-cases.json
python -m pytest tests/test_add.py -m add --device cpu --ref-as-candidate
python -m pytest tests/test_add.py benchmark/test_add_benchmark.py -m add --device cpu --dummy-candidate add --dummy-report add-dummy.json
python -m pytest tests/test_add.py -m add --override add:/path/to/candidate.py
python -m pytest benchmark/test_add_benchmark.py -m add --override add:/path/to/candidate.py \
  --case-id 'benchmark/test_add_benchmark.py::test_add_benchmark::core::uint8::3' \
  --record json --output add-result.json
```

Append `:FUNCTION` to the override specification if the candidate entry point
has another name. The runtime reads `GEMS_VENDOR`, otherwise detects the accelerator automatically.
Use `--device cpu` to run CPU checks or an indexed device such as `npu:1` to
select a device explicitly. Unknown vendors and conflicting accelerator device
types raise an error. Use `--reference FILE:FUNCTION`
to additionally check an independent oracle with the same signature. This
compares the selected reference and candidate against the supplied oracle.
Correctness and `--ref-as-candidate` select the reference by operator and
device type. Register backend-specific references in `reference/`.
When there is no backend-specific entry, the runner uses the operator's
default reference.

Performance comparisons default to `--baseline opencv-cuda`. Each case uses the
same source tensors with independent preallocated outputs. The runner validates
candidate and OpenCV outputs against the Torch reference and each other before
sampling, and checks that inputs are unchanged. It alternates the measurement
order over `--benchmark-samples` paired samples and reports median latencies.
OpenCV errors and missing bridge libraries fail the case. Use `--baseline none`
explicitly for candidate-only timing; other backends need this mode.

With `--record json`, benchmark output uses per-operator `details` and per-dtype
`result` rows. `latency`, `base_latency_ms` and their sample arrays are in
milliseconds. `base_raw_speedup` and `compared_speedup` are both live baseline
latency divided by candidate latency; `baseline_source` is `opencv_cuda_live`
and `baseline_version` identifies the linked OpenCV version. Disabled baselines
have null baseline latency and speedup fields. Historical `--base-data` JSON
lookup is no longer used.

Correctness JSON records include pytest `params`, an `opname` list, and
per-test `candidate_calls`. The add report stores `params.shape` as an array
and `params.dtype` as a string. Reference
validation also records `reference_calls` and `reference_candidate_calls`.
Dummy evidence is written separately to `--dummy-report` (default
`dummy-candidate-report.json`) using `flaggems.dummy-candidate/v1`.
Each correctness case and benchmark workload must reach the candidate.
Timing results identify `candidate_source` only after the timer invokes it.

For compatibility with FlagGems-sglang, `--override` accepts repeated
`OP:FILE[:FUNCTION]` arguments and the `OP=FILE[:FUNCTION]` spelling.
`--record json --output FILE` enables result JSON. Case listing always
writes JSON. Profile, dummy, reference mode and listing are mutually
exclusive. Duplicate or unknown benchmark case IDs fail explicitly.
Tests, benchmark and profile use the shared `runtime.resolve_device` result.
Set `GEMS_VENDOR` for an explicit vendor selection or leave it unset for
automatic detection. `--device` preserves explicit device selection.

### Profile one candidate case

Pass a case ID from `--list-cases` to replay just that workload:

```sh
python -m pytest benchmark/test_add_benchmark.py --profile-only \
  --case-id 'benchmark/test_add_benchmark.py::test_add_benchmark::core::uint8::3' \
  --override add:/path/to/candidate.py --profile-warmup 10 --profile-iterations 1
```

The runner constructs inputs and warms up before capture. A pytest plugin can
implement `pytest_flaggems_profile_scope(backend, case_id)` and return a context
manager around the candidate calls and final synchronization. Without the
plugin, the command only replays the candidate. Profile mode does not run
correctness, collect benchmark latency, or read performance baseline data.
