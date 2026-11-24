/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <flagcv/flagcv.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "reference/cpu/pointwise.hpp"

namespace {
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
  void* data = nullptr;
  explicit Buffer(std::size_t bytes) { cuda_check(cudaMalloc(&data, bytes)); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
struct Stream {
  cudaStream_t value = nullptr;
  Stream() {
    cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
  }
  ~Stream() { cudaStreamDestroy(value); }
};
const char* names[] = {"add", "subtract", "multiply", "bitwise_and",
                       "bitwise_not"};
void invoke(int mode, flagcv::Handle& handle, const flagcv::GpuMat& a,
            const flagcv::GpuMat& b, const flagcv::GpuMat& dst,
            cudaStream_t stream) {
  switch (mode) {
    case 0:
      flagcv::add(handle, a, b, dst, stream);
      break;
    case 1:
      flagcv::subtract(handle, a, b, dst, stream);
      break;
    case 2:
      flagcv::multiply(handle, a, b, dst, stream);
      break;
    case 3:
      flagcv::bitwise_and(handle, a, b, dst, stream);
      break;
    case 4:
      flagcv::bitwise_not(handle, a, dst, stream);
      break;
  }
}
int checks = 0;
void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

template <class T>
void run_case(flagcv::Handle& handle, int mode, int width, int height,
              int channels, bool padded, int alias, cudaStream_t stream) {
  constexpr bool bytes = std::is_integral_v<T>;
  constexpr flagcv::Depth depth =
      bytes ? flagcv::Depth::k8U : flagcv::Depth::k32F;
  const int row = width * channels;
  const int pitch_a = row + (padded ? 7 : 0);
  const int pitch_b = row + (padded ? 11 : 0);
  const int pitch_d = alias == 1   ? pitch_a
                      : alias == 2 ? pitch_b
                                   : row + (padded ? 13 : 0);
  const int offset = padded ? 1 : 0;
  const T sentinel = static_cast<T>(bytes ? 165 : -123);
  std::vector<T> a(offset + pitch_a * height + 3, sentinel);
  std::vector<T> b(offset + pitch_b * height + 3, sentinel);
  std::vector<T> d(offset + pitch_d * height + 3, sentinel);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < row; ++x) {
      const int i = y * row + x;
      if constexpr (bytes) {
        // Includes 0, 1, 127, 128, 254, 255 and nonzero under/overflow pairs.
        constexpr int edges[] = {0, 1, 127, 128, 254, 255, 3};
        a[offset + y * pitch_a + x] = static_cast<T>(edges[i % 7]);
        b[offset + y * pitch_b + x] = static_cast<T>(edges[(i * 3 + 5) % 7]);
      } else {
        a[offset + y * pitch_a + x] = static_cast<T>((i % 509 - 254) * 0.137);
        b[offset + y * pitch_b + x] = static_cast<T>((i % 127 - 63) * 0.0314);
      }
    }
  }
  std::vector<T> expected = alias == 1 ? a : alias == 2 ? b : d;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < row; ++x)
      expected[offset + y * pitch_d + x] = flagcv::reference::pointwise(
          mode, a[offset + y * pitch_a + x], b[offset + y * pitch_b + x]);
  Buffer da(a.size() * sizeof(T)), db(b.size() * sizeof(T)),
      dd(d.size() * sizeof(T));
  cuda_check(cudaMemcpyAsync(da.data, a.data(), a.size() * sizeof(T),
                             cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemcpyAsync(db.data, b.data(), b.size() * sizeof(T),
                             cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemcpyAsync(dd.data, d.data(), d.size() * sizeof(T),
                             cudaMemcpyHostToDevice, stream));
  void* out = alias == 1 ? da.data : alias == 2 ? db.data : dd.data;
  flagcv::GpuMat ga(static_cast<T*>(da.data) + offset, width, height, depth,
                    channels, pitch_a * sizeof(T));
  flagcv::GpuMat gb(static_cast<T*>(db.data) + offset, width, height, depth,
                    channels, pitch_b * sizeof(T));
  flagcv::GpuMat gd(static_cast<T*>(out) + offset, width, height, depth,
                    channels, pitch_d * sizeof(T));
  invoke(mode, handle, ga, gb, gd, stream);
  std::vector<T> actual(expected.size()), after_a(a.size()), after_b(b.size());
  cuda_check(cudaMemcpyAsync(actual.data(), out, actual.size() * sizeof(T),
                             cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaMemcpyAsync(after_a.data(), da.data, a.size() * sizeof(T),
                             cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaMemcpyAsync(after_b.data(), db.data, b.size() * sizeof(T),
                             cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  const std::string context = std::string(names[mode]) +
                              (bytes ? " 8U" : " 32F") + " C" +
                              std::to_string(channels);
  require(actual == expected, context + " result/padding mismatch");
  if (alias != 1) require(after_a == a, context + " modified src1");
  if (alias != 2) require(after_b == b, context + " modified src2");
}

template <class F>
void rejects(F call, const std::string& name) {
  bool rejected = false;
  try {
    call();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "did not reject " + name);
}

void validation(flagcv::Handle& h) {
  using flagcv::Depth;
  using flagcv::GpuMat;
  Buffer buffer(65536);
  char* base = static_cast<char*>(buffer.data);
  GpuMat a(base, 16, 8, Depth::k8U, 1);
  GpuMat b(base + 4096, 16, 8, Depth::k8U, 1);
  GpuMat d(base + 8192, 16, 8, Depth::k8U, 1);
  for (int mode = 0; mode < 5; ++mode) {
    rejects([&] { invoke(mode, h, GpuMat(), b, d, nullptr); }, "empty input");
    rejects(
        [&] {
          invoke(mode, h, a, b, GpuMat(base + 8192, 15, 8, Depth::k8U, 1),
                 nullptr);
        },
        "extent mismatch");
    rejects(
        [&] {
          invoke(mode, h, a, b, GpuMat(base + 8192, 16, 8, Depth::k8U, 2),
                 nullptr);
        },
        "channel mismatch");
    rejects(
        [&] {
          invoke(mode, h, a, b, GpuMat(base + 8192, 16, 8, Depth::k32F, 1),
                 nullptr);
        },
        "depth mismatch");
    rejects(
        [&] {
          invoke(mode, h, a, b, GpuMat(base + 1, 16, 8, Depth::k8U, 1),
                 nullptr);
        },
        "partial overlap");
    rejects(
        [&] {
          invoke(mode, h, a, b, GpuMat(base, 16, 8, Depth::k8U, 1, 32),
                 nullptr);
        },
        "overlap with unequal pitches");
    rejects(
        [&] {
          invoke(mode, h, a, b, GpuMat(base + 8192, 16, 8, Depth::k8U, 1, 8),
                 nullptr);
        },
        "short pitch");
    for (Depth depth : {Depth::k16U, Depth::k32F}) {
      if (depth == Depth::k32F && mode < 3) continue;
      rejects(
          [&] {
            invoke(mode, h, GpuMat(base, 16, 8, depth, 1),
                   GpuMat(base + 4096, 16, 8, depth, 1),
                   GpuMat(base + 8192, 16, 8, depth, 1), nullptr);
          },
          "unsupported depth");
    }
  }
}

// CUDA graph timing excludes compilation and per-call CPU submission overhead.
void benchmark(flagcv::Handle& handle, int mode, int width, int height,
               flagcv::Depth depth) {
  const std::size_t bytes = std::size_t{static_cast<unsigned>(width)} * height *
                            flagcv::depth_size(depth);
  Buffer a(bytes), b(bytes), d(bytes);
  Stream stream;
  cuda_check(cudaMemsetAsync(a.data, 0, bytes, stream.value));
  cuda_check(cudaMemsetAsync(b.data, 0, bytes, stream.value));
  flagcv::GpuMat ga(a.data, width, height, depth, 1),
      gb(b.data, width, height, depth, 1), gd(d.data, width, height, depth, 1);
  invoke(mode, handle, ga, gb, gd, stream.value);
  cuda_check(cudaStreamSynchronize(stream.value));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  cuda_check(
      cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
  constexpr int iterations = 100;
  for (int i = 0; i < iterations; ++i)
    invoke(mode, handle, ga, gb, gd, stream.value);
  cuda_check(cudaStreamEndCapture(stream.value, &graph));
  cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  cudaEvent_t start, stop;
  cuda_check(cudaEventCreate(&start));
  cuda_check(cudaEventCreate(&stop));
  cuda_check(cudaGraphLaunch(exec, stream.value));
  cuda_check(cudaEventRecord(start, stream.value));
  constexpr int repeats = 10;
  for (int i = 0; i < repeats; ++i)
    cuda_check(cudaGraphLaunch(exec, stream.value));
  cuda_check(cudaEventRecord(stop, stream.value));
  cuda_check(cudaEventSynchronize(stop));
  float elapsed = 0;
  cuda_check(cudaEventElapsedTime(&elapsed, start, stop));
  std::printf("perf,%s,%dx%d,%s,%.3f us\n", names[mode], height, width,
              depth == flagcv::Depth::k8U ? "uint8" : "float32",
              elapsed * 1000 / (iterations * repeats));
  cuda_check(cudaEventDestroy(start));
  cuda_check(cudaEventDestroy(stop));
  cuda_check(cudaGraphExecDestroy(exec));
  cuda_check(cudaGraphDestroy(graph));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    flagcv::Handle handle("nvidia", 0);
    if (argc == 2 && std::string(argv[1]) == "--benchmark") {
      for (int mode = 0; mode < 5; ++mode)
        for (auto depth : {flagcv::Depth::k8U, flagcv::Depth::k32F}) {
          if (mode >= 3 && depth != flagcv::Depth::k8U) continue;
          for (auto shape : {std::pair{64, 64}, std::pair{640, 480},
                             std::pair{1920, 1080}, std::pair{3840, 2160}})
            benchmark(handle, mode, shape.first, shape.second, depth);
        }
      return 0;
    }
    Stream stream;
    for (int mode = 0; mode < 5; ++mode) {
      for (int channels : {1, 2, 3, 4}) {
        for (bool padded : {false, true}) {
          for (int alias : {0, 1, 2}) {
            run_case<std::uint8_t>(handle, mode, 67, 61, channels, padded,
                                   alias, stream.value);
            if (mode < 3)
              run_case<float>(handle, mode, 67, 61, channels, padded, alias,
                              stream.value);
          }
        }
      }
      for (auto shape : {std::pair{1, 1}, std::pair{1, 257}, std::pair{257, 1},
                         std::pair{1027, 5}, std::pair{1920, 1080}}) {
        run_case<std::uint8_t>(handle, mode, shape.first, shape.second, 1,
                               false, 0, nullptr);
        if (mode < 3)
          run_case<float>(handle, mode, shape.first, shape.second, 1, false, 0,
                          nullptr);
      }
    }
    validation(handle);
    std::printf("%d checks passed\n", checks);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
