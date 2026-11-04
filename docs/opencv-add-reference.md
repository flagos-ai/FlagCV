# OpenCV CUDA add reference

Build the CUDA-enabled OpenCV library and the shared bridge using the
[ops-baseline OpenCV instructions](https://github.com/zhaojs12345/ops-baseline/tree/3b2f022314b0d6e3bf6857f400c31beb65a505dd/ops/opencv).
Set `OPENCV_BASELINE_LIBRARY` to the built shared library and
`OPENCV_BASELINE_RECIPE` to the absolute path of `ops/opencv/add.py`.

The adapter exports `add(a, b, dst)`. It takes contiguous single-channel
CUDA tensors of matching float32 or uint8 dtype with a separate preallocated
destination. It reuses the recipe's device and stream handling.

With the Python evaluation from FlagCV PR #4 available, use the adapter as
the independent correctness oracle:

```sh
python -m pytest tests/test_add.py --override add:/path/to/candidate.py \
  --reference adapters/opencv_add.py:add
```

Performance data is collected by ops-baseline and supplied to the evaluation
benchmark with `--base-data`.
