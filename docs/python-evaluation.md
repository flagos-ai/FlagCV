# Python kernel evaluation

FlagCV's Python evaluation workflow supports case listing and selection,
candidate override, dummy candidate injection, reference as candidate, and
benchmark comparison with static baseline data. Operator tests and benchmarks
use the root pytest options to expose these capabilities. Each operator defines
its own callable signature, inputs, and reference.

The Python Definition describes the candidate interface. The C++ integration
has its own CTest entry points.

Register an operator with `definitions/OP.json` and mark its evaluation tests
and benchmarks with `@pytest.mark.OP`. Define its default reference as
`reference/OP.py` with a callable named `reference`. The runner uses the marker
to select the reference or the matching `--override` entry. Dummy mode selects
only the operator named by `--dummy-candidate`.

## Example: add

The `add` profile matches
[ops-baseline's OpenCV CUDA add](https://github.com/zhaojs12345/ops-baseline/tree/3b2f022314b0d6e3bf6857f400c31beb65a505dd/ops/opencv).
It covers four image sizes with float32 and uint8, contiguous single-channel
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
  --case-id 'benchmark/test_add_benchmark.py::test_add_benchmark::core::uint8::3' --base-data /path/to/op_opencv_perf_baseline.json \
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

For this profile, baseline lookup requires an exact shape, dtype and
OpenCV profile match. With `--record json`, benchmark output uses per-operator `details` and
per-dtype `result` rows. `latency` is in milliseconds, and
`raw_speedup` is baseline latency divided by candidate latency. The
OpenCV-specific baseline fields retain their raw latency-ratio meaning.
Correctness JSON records include per-test `candidate_calls`. Reference
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
