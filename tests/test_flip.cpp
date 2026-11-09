/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

/*
 * flip through the full runtime path: lowering, kernel resolution, the
 * external compiler process, the nvidia plugin and launch.
 *
 * Device memory comes from the CUDA runtime because FlagCV never allocates.
 * Every check is class A (bit-exact) per docs/numerical-tolerance.md, and
 * also verifies that row padding in dst is never written.
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <flagcv/flagcv.hpp>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "reference/cpu/flip.hpp"
#include "tests/common/tolerance.hpp"

namespace {

int failures = 0;
int checks = 0;

#define CUDA_OK(call)                                                    \
  do {                                                                   \
    cudaError_t err_ = (call);                                           \
    if (err_ != cudaSuccess) {                                           \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(err_));                            \
      std::exit(2);                                                      \
    }                                                                    \
  } while (0)

void expect(bool ok, const std::string& what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
  }
}

const char* depth_name(flagcv::Depth depth) {
  switch (depth) {
    case flagcv::Depth::k8U:
      return "8U";
    case flagcv::Depth::k8S:
      return "8S";
    case flagcv::Depth::k16U:
      return "16U";
    case flagcv::Depth::k16S:
      return "16S";
    case flagcv::Depth::k32S:
      return "32S";
    case flagcv::Depth::k16F:
      return "16F";
    case flagcv::Depth::k32F:
      return "32F";
    case flagcv::Depth::k64F:
      return "64F";
  }
  return "?";
}

constexpr std::uint8_t kSentinel = 0xA5;

struct Case {
  int width, height;
  flagcv::Depth depth;
  int channels;
  int flip_code;
  std::int64_t src_pad = 0, dst_pad = 0;  // extra bytes per row
  std::size_t src_offset = 0;             // bytes past the allocation start
  std::size_t dst_offset = 0;

  std::string name() const {
    return std::to_string(height) + "x" + std::to_string(width) + " " +
           depth_name(depth) + "C" + std::to_string(channels) +
           " code=" + std::to_string(flip_code) +
           " pad=" + std::to_string(src_pad) + "/" + std::to_string(dst_pad) +
           " off=" + std::to_string(src_offset) + "/" +
           std::to_string(dst_offset);
  }
};

/* Runs one case against the CPU reference over the whole dst allocation,
 * padding and offset bytes included. */
void run_case(flagcv::Handle& handle, const Case& c,
              cudaStream_t stream = nullptr) {
  const std::size_t pixel = flagcv::depth_size(c.depth) * c.channels;
  const std::int64_t src_pitch = c.width * pixel + c.src_pad;
  const std::int64_t dst_pitch = c.width * pixel + c.dst_pad;
  const std::size_t src_bytes = c.src_offset + src_pitch * c.height;
  const std::size_t dst_bytes = c.dst_offset + dst_pitch * c.height;

  std::vector<std::uint8_t> host_src(src_bytes), expected(dst_bytes, kSentinel),
      actual(dst_bytes);
  std::mt19937 rng(
      static_cast<unsigned>(c.width * 131 + c.height * 7 + c.channels));
  for (auto& b : host_src) b = static_cast<std::uint8_t>(rng());
  flagcv::reference::flip(host_src.data() + c.src_offset, src_pitch,
                          expected.data() + c.dst_offset, dst_pitch, c.width,
                          c.height, pixel, c.flip_code);

  void *d_src = nullptr, *d_dst = nullptr;
  CUDA_OK(cudaMalloc(&d_src, src_bytes));
  CUDA_OK(cudaMalloc(&d_dst, dst_bytes));
  CUDA_OK(
      cudaMemcpy(d_src, host_src.data(), src_bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(d_dst, kSentinel, dst_bytes));

  flagcv::GpuMat src(static_cast<char*>(d_src) + c.src_offset, c.width,
                     c.height, c.depth, c.channels, c.src_pad ? src_pitch : 0);
  flagcv::GpuMat dst(static_cast<char*>(d_dst) + c.dst_offset, c.width,
                     c.height, c.depth, c.channels, c.dst_pad ? dst_pitch : 0);
  try {
    flagcv::flip(handle, src, dst, c.flip_code, stream);
    CUDA_OK(stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize());
    CUDA_OK(
        cudaMemcpy(actual.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost));
    std::string diff =
        flagcv::testing::bit_exact(actual.data(), expected.data(), dst_bytes);
    expect(diff.empty(), c.name() + ": " + diff);
  } catch (const std::exception& e) {
    expect(false, c.name() + ": threw " + e.what());
  }
  CUDA_OK(cudaFree(d_src));
  CUDA_OK(cudaFree(d_dst));
}

void expect_rejected(const std::string& what,
                     const std::function<void()>& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  expect(rejected, "rejects " + what);
}

double elapsed_ms(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - since)
      .count();
}

void report_performance(flagcv::Handle& handle, int width, int height,
                        flagcv::Depth depth, int channels, int flip_code) {
  const std::size_t bytes = static_cast<std::size_t>(width) * height *
                            channels * flagcv::depth_size(depth);
  void *d_src = nullptr, *d_dst = nullptr;
  CUDA_OK(cudaMalloc(&d_src, bytes));
  CUDA_OK(cudaMalloc(&d_dst, bytes));
  flagcv::GpuMat src(d_src, width, height, depth, channels),
      dst(d_dst, width, height, depth, channels);

  cudaEvent_t start, stop;
  CUDA_OK(cudaEventCreate(&start));
  CUDA_OK(cudaEventCreate(&stop));
  auto time = [&](const std::function<void()>& body) {
    for (int i = 0; i < 10; ++i) body();
    CUDA_OK(cudaEventRecord(start));
    const int iterations = 200;
    for (int i = 0; i < iterations; ++i) body();
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    return ms / iterations;
  };
  const double flip_ms =
      time([&] { flagcv::flip(handle, src, dst, flip_code); });
  const double copy_ms = time([&] {
    CUDA_OK(cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
  });
  auto gbps = [&](double ms) { return 2.0 * bytes / (ms * 1e-3) / 1e9; };
  std::printf(
      "perf %5dx%-5d %3sC%d code=%2d: flip %.4f ms (%4.0f GB/s), D2D memcpy "
      "%.4f ms (%4.0f GB/s), "
      "flip/memcpy speed %.2f\n",
      height, width, depth_name(depth), channels, flip_code, flip_ms,
      gbps(flip_ms), copy_ms, gbps(copy_ms), copy_ms / flip_ms);
  CUDA_OK(cudaEventDestroy(start));
  CUDA_OK(cudaEventDestroy(stop));
  CUDA_OK(cudaFree(d_src));
  CUDA_OK(cudaFree(d_dst));
}

}  // namespace

int main() {
  using flagcv::Depth;

  auto t0 = std::chrono::steady_clock::now();
  flagcv::Handle handle("nvidia", 0);
  std::printf("handle created in %.1f ms\n", elapsed_ms(t0));

  // First call pays for starting the compiler process and one compile; the
  // second, same layout and a different size, only launches.
  {
    Case c{640, 480, Depth::k8U, 3, 1};
    t0 = std::chrono::steady_clock::now();
    run_case(handle, c);
    std::printf("first flip (compiler start + compile + launch): %.1f ms\n",
                elapsed_ms(t0));
    c.width = 320;
    c.height = 200;
    t0 = std::chrono::steady_clock::now();
    run_case(handle, c);
    std::printf("second flip, new size, same kernel (launch only): %.1f ms\n",
                elapsed_ms(t0));
  }

  // Every flip direction across depths and channel counts, with padded and
  // unequal pitches. Odd extents exercise the tile edges.
  t0 = std::chrono::steady_clock::now();
  for (Depth depth : {Depth::k8U, Depth::k16U, Depth::k32F, Depth::k64F}) {
    for (int channels : {1, 3, 4}) {
      for (int code : {0, 1, -1}) {
        run_case(handle, Case{53, 37, depth, channels, code, 64, 32});
      }
    }
  }
  std::printf("depth x channels x direction sweep (36 kernels): %.1f ms\n",
              elapsed_ms(t0));

  // Remaining depths: flip moves bits, so signed and half types must too.
  for (Depth depth : {Depth::k8S, Depth::k16S, Depth::k32S, Depth::k16F}) {
    run_case(handle, Case{29, 17, depth, 2, -1});
  }

  // Degenerate and boundary extents.
  for (int code : {0, 1, -1}) {
    run_case(handle, Case{1, 1, Depth::k8U, 1, code});
    run_case(handle, Case{7, 1, Depth::k8U, 1, code});
    run_case(handle, Case{1, 9, Depth::k32F, 3, code});
    run_case(handle,
             Case{1027, 5, Depth::k8U, 1, code});  // spans two BLOCK_X tiles
    run_case(handle, Case{1920, 1080, Depth::k8U, 3, code});
  }

  // Base pointers aligned to the depth but not to 16 bytes compile a second,
  // unhinted kernel; both must be correct.
  run_case(handle, Case{61, 13, Depth::k8U, 1, 1, 0, 0, 1, 3});
  run_case(handle, Case{61, 13, Depth::k32F, 3, -1, 4, 0, 4, 8});

  // A caller stream.
  {
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    run_case(handle, Case{300, 200, Depth::k16U, 4, 0, 16, 0}, stream);
    CUDA_OK(cudaStreamDestroy(stream));
  }

  // Validation happens in lowering, before any compile or launch.
  {
    void* buffer = nullptr;
    CUDA_OK(cudaMalloc(&buffer, 1 << 20));
    char* base = static_cast<char*>(buffer);
    using flagcv::GpuMat;
    GpuMat src(base, 64, 32, Depth::k8U, 1);
    GpuMat dst(base + 65536, 64, 32, Depth::k8U, 1);
    expect_rejected("empty src",
                    [&] { flagcv::flip(handle, GpuMat(), dst, 0); });
    expect_rejected("extent mismatch", [&] {
      flagcv::flip(handle, src, GpuMat(base + 65536, 63, 32, Depth::k8U, 1), 0);
    });
    expect_rejected("depth mismatch", [&] {
      flagcv::flip(handle, src, GpuMat(base + 65536, 64, 32, Depth::k16U, 1),
                   0);
    });
    expect_rejected("channel mismatch", [&] {
      flagcv::flip(handle, src, GpuMat(base + 65536, 64, 32, Depth::k8U, 3), 0);
    });
    expect_rejected("in place", [&] { flagcv::flip(handle, src, src, 0); });
    expect_rejected("partial overlap", [&] {
      flagcv::flip(handle, src, GpuMat(base + 100, 64, 32, Depth::k8U, 1), 0);
    });
    expect_rejected("pitch shorter than a row", [&] {
      flagcv::flip(handle, GpuMat(base, 64, 32, Depth::k8U, 1, 32), dst, 0);
    });
    expect_rejected("five channels", [&] {
      flagcv::flip(handle, GpuMat(base, 8, 8, Depth::k8U, 5),
                   GpuMat(base + 65536, 8, 8, Depth::k8U, 5), 0);
    });
    expect_rejected("pointer misaligned for depth", [&] {
      flagcv::flip(handle, GpuMat(base + 1, 8, 8, Depth::k16U, 1),
                   GpuMat(base + 65536, 8, 8, Depth::k16U, 1), 0);
    });
    expect_rejected("non-positive width", [&] {
      flagcv::flip(handle, GpuMat(base, 0, 8, Depth::k8U, 1),
                   GpuMat(base + 65536, 0, 8, Depth::k8U, 1), 0);
    });
    CUDA_OK(cudaFree(buffer));
  }

  // Timing is informational, not asserted. 4K frames fit in L2 on recent
  // GPUs; the 128 MiB image measures DRAM bandwidth.
  for (int code : {0, -1}) {
    report_performance(handle, 3840, 2160, Depth::k8U, 1, code);
    report_performance(handle, 3840, 2160, Depth::k8U, 3, code);
    report_performance(handle, 3840, 2160, Depth::k32F, 1, code);
    report_performance(handle, 16384, 8192, Depth::k8U, 1, code);
  }

  std::printf("%d/%d checks passed\n", checks - failures, checks);
  return failures == 0 ? 0 : 1;
}
