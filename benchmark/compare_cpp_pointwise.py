"""Compare the C++ runtime's CUDA-graph timings with the Torch references.

Example (build with FLAGCV_BUILD_TESTS=ON first):
  CUDA_VISIBLE_DEVICES=0 python benchmark/compare_cpp_pointwise.py \
      --executable build-cpp-operators/tests/test_pointwise --output results.json

These are steady-state GPU timings, excluding compilation and CPU submission.
The baseline is reference/*.py, not OpenCV CUDA.
"""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Run C++ first so the two processes do not benchmark concurrently.
    output = subprocess.check_output(
        [str(args.executable.resolve()), "--benchmark"], text=True
    )

    import torch
    from triton.testing import do_bench_cudagraph

    root = Path(__file__).resolve().parents[1]
    references = {}
    rows = []
    for line in output.splitlines():
        if not line.startswith("perf,"):
            continue
        _, op, shape_text, dtype, cpp_text = line.split(",")
        shape = tuple(map(int, shape_text.split("x")))
        cpp_us = float(cpp_text.removesuffix(" us"))
        if op not in references:
            spec = importlib.util.spec_from_file_location(
                f"flagcv_reference_{op}", root / "reference" / f"{op}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            references[op] = module.reference
        a = torch.zeros(shape, device="cuda", dtype=getattr(torch, dtype))
        b, dst = torch.zeros_like(a), torch.empty_like(a)
        operands = (a, dst) if op == "bitwise_not" else (a, b, dst)
        reference = references[op]
        reference_us = 1000 * do_bench_cudagraph(
            lambda: reference(*operands), rep=20, n_retries=3
        )
        rows.append(
            dict(
                operator=op,
                shape=shape,
                dtype=dtype,
                cpp_us=cpp_us,
                torch_reference_us=reference_us,
                speedup=reference_us / cpp_us,
            )
        )
    if not rows:
        raise RuntimeError("C++ benchmark produced no performance records")
    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        timing="CUDA graph, steady-state GPU time in microseconds",
        baseline="reference/*.py (Torch)",
        records=rows,
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
