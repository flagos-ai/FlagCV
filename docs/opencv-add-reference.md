# OpenCV CUDA reference

The benchmark runner executes OpenCV CUDA in the same process as the candidate.
The native bridge wraps caller-owned device pointers as `cv::cuda::GpuMat`,
uses a separate preallocated output, and enqueues the OpenCV operation on the
current PyTorch CUDA stream. Input data stays on the device. The bridge is a
validation dependency; the production FlagCV library does not link OpenCV.

## Build the bridge

Use an OpenCV build with CUDA and the contrib `cudaarithm` module, compiled for
the target GPU. A CPU-only OpenCV installation or Python wheel is insufficient.
The bridge links the C++ library directly and does not import `cv2`.

```sh
cmake -S adapters/opencv_cuda -B build-opencv-cuda \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
cmake --build build-opencv-cuda -j 8
export FLAGCV_OPENCV_LIBRARY="$PWD/build-opencv-cuda/libflagcv_opencv_cuda.so"
```

`OpenCV_DIR` is the directory containing `OpenCVConfig.cmake`; its path depends
on the OpenCV version and installation layout. If the environment variable is
unset, the adapter looks for `build-opencv-cuda/libflagcv_opencv_cuda.so` under
the FlagCV source root.

The bridge accepts matching contiguous two-dimensional, single-channel CUDA
tensors with uint8 or float32 dtype; bitwise inputs use uint8. Arithmetic uses
matching output depth and unit multiply scale. Flip uses OpenCV's direction
convention. Sources and output are owned by the caller; native GpuMat headers
and stream wrappers are prepared before timing.

## Run a live comparison

```sh
CUDA_VISIBLE_DEVICES=0 python -m pytest benchmark/test_subtract_benchmark.py \
  --device cuda:0 --override subtract:kernels/common/subtract.py \
  --record json --output subtract-result.json
```

The default `--baseline opencv-cuda` requires the NVIDIA backend and an available
bridge. It validates both outputs against the operator's Torch reference and
each other, checks input preservation, then alternates candidate and OpenCV
measurements with the same Triton event timer. Failures are reported directly;
there is no CPU or historical-data fallback. Inputs and outputs are allocated
before timing, and first-call compilation is outside timed samples.

Use `--benchmark-samples` to set the number of paired measurements. The reported
latencies are medians of the samples. They describe steady-state device work;
build time, CPU submission time and transfers are not separate reported metrics.
Use `--baseline none` explicitly for candidate-only timing on another backend.
Case listing, dummy probes and profile mode do not load the OpenCV bridge.

## Optional independent accuracy reference for add

The older `adapters/opencv_add.py` adapter remains available for the separate
ops-baseline bridge. Set `OPENCV_BASELINE_LIBRARY` to that bridge library and
`OPENCV_BASELINE_RECIPE` to its `ops/opencv/add.py` recipe, then run:

```sh
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_add.py \
  --override add:/path/to/candidate.py --reference adapters/opencv_add.py:add
```

Those environment variables apply only to the older adapter. The live
performance comparison uses `FLAGCV_OPENCV_LIBRARY` and does not read historical
performance JSON files.
