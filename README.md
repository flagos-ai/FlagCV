# FlagCV

## About

FlagCV is part of FlagOS, a fully open-source system software stack designed to unify the model–system–chip layers and foster an open and collaborative ecosystem. It enables a "develop once, run anywhere" workflow across diverse AI accelerators, unlocking hardware performance, eliminating fragmentation among AI chipset-specific software stacks, and substantially lowering the cost of porting and maintaining AI workloads.

FlagCV is a high-performance computer vision library implemented in Triton language. It provides optimized implementations of common image processing primitives (color conversion, filtering, geometric transforms, and arithmetic operations) behind OpenCV-shaped interfaces, benchmarked against OpenCV CUDA baselines to deliver reference-level correctness with competitive performance across diverse GPU architectures.

Built on FlagTree (a FlagOS-maintained Triton fork supporting multiple hardware backends), FlagCV offers a vendor-agnostic operator interface with pluggable backend support.

## Quick Start

### Requirements

Common to every backend:

- CMake 3.23+ and a C++17 compiler
- Python 3 with [Triton/FlagTree](https://github.com/FlagTree/flagtree) installed. FlagCV compiles kernels at run time in a Python process that it starts itself; `libflagcv` does not link Python.

Plus the toolchain of the backend you build. `libflagcv` itself links no vendor SDK and no CUDA: it reaches the device only through a `dlopen`'d plugin, so selecting a backend never changes the core.

| `FLAGCV_BACKENDS` entry | Plugin links |
|---|---|
| `nvidia` | CUDA driver API (`CUDA::cuda_driver`) |
| `iluvatar` | CUDA-compatible driver API (`CUDA::cuda_driver`) |
| `ascend` | AscendCL / runtime |
| `hygon` | HIP runtime |
| `mthreads` | MUSA runtime |
| `thead` | PPU driver API (`CUstream`), T-Head PPU SDK |

The NVIDIA backend is the one built and tested here; the other five implement
the same plugin contract and compile against their SDKs when it is present
(`-DCUDAToolkit_ROOT=<corex>` for `iluvatar`).

### Build and test

```sh
cmake -B build -DBUILD_SHARED_LIBS=ON -DFLAGCV_BUILD_TESTS=ON
cmake --build build -j
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure
```

`FLAGCV_BUILD_TESTS` builds operator tests for the backends a device runtime is wired up for.

### Use from C++

Install, then point your project at the install prefix:

```sh
cmake --install build --prefix /opt/flagcv
```

Build the library as shared when installing it: `libflagcv` finds its backend plugins and kernel sources relative to its own location.

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.23)
project(flip_demo LANGUAGES CXX)

find_package(FlagCV REQUIRED)
find_package(CUDAToolkit REQUIRED)  # only for cudaMalloc in this demo

add_executable(flip_demo main.cpp)
target_link_libraries(flip_demo PRIVATE FlagCV::flagcv CUDA::cudart)
```

`main.cpp`:

```cpp
#include <flagcv/flagcv.hpp>

#include <cuda_runtime.h>

#include <cstdio>

int main() {
  const int width = 4, height = 2;
  unsigned char host[] = {1, 2, 3, 4,
                          5, 6, 7, 8};

  // FlagCV never allocates device memory: the caller owns both buffers.
  void *src = nullptr, *dst = nullptr;
  cudaMalloc(&src, sizeof(host));
  cudaMalloc(&dst, sizeof(host));
  cudaMemcpy(src, host, sizeof(host), cudaMemcpyHostToDevice);

  flagcv::Handle handle("nvidia", 0);  // backend plugin + kernel cache
  flagcv::GpuMat in(src, width, height, flagcv::Depth::k8U, 1);
  flagcv::GpuMat out(dst, width, height, flagcv::Depth::k8U, 1);
  flagcv::flip(handle, in, out, 1);  // 1: mirror columns, 0: rows, -1: both

  // flip is enqueued on the default stream; this copy waits for it.
  cudaMemcpy(host, dst, sizeof(host), cudaMemcpyDeviceToHost);
  for (unsigned char v : host) std::printf("%d ", v);  // 4 3 2 1 8 7 6 5
  std::printf("\n");

  cudaFree(src);
  cudaFree(dst);
}
```

```sh
cmake -B build -DCMAKE_PREFIX_PATH=/opt/flagcv
cmake --build build
./build/flip_demo
```

The first call in a process takes about 2 seconds: it starts the compiler process and compiles the kernel. Later calls only launch. A call compiles again only when it needs a new kernel variant: a different pixel depth, channel count or flip direction, or a width, row pitch or buffer address that changes between a multiple of 16 and not. Each variant compiles once per process. Compiled binaries are cached under `~/.cache/flagcv`.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `FLAGCV_PYTHON` | `python3` | Interpreter for the compiler process; it must have Triton installed |
| `FLAGCV_CACHE_DIR` | `~/.cache/flagcv` | Where compiled kernel binaries are written |
| `FLAGCV_DATA_DIR` | `<prefix>/share/flagcv`, else the source tree | Kernel sources, compiler and platform providers |
| `FLAGCV_PLUGIN_DIR` | `<prefix>/lib/flagcv`, else the build tree | Backend plugins (`libflagcv_<backend>.so`) |

See [docs/architecture.md](docs/architecture.md) for how the runtime, the compiler process and the plugins fit together, and [docs/operator-development.md](docs/operator-development.md) for adding an operator.
