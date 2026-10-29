/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagcv/flagcv.hpp>

#include <hip/hip_runtime_api.h>

#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
  try {
    int previous = -1;
    require(hipGetDevice(&previous) == hipSuccess, "cannot query HIP device");
    {
      flagcv::Handle handle("amd", 0);
      require(handle.backend() == "amd", "incorrect backend");
      flagcv::Handle moved(std::move(handle));
      flagcv::Handle assigned("amd", 0);
      assigned = std::move(moved);
      require(assigned.backend() == "amd", "move lost backend");
    }
    int current = -1;
    require(hipGetDevice(&current) == hipSuccess && current == previous,
            "Handle changed the caller's HIP device");
    for (const auto* backend : {"amd", "../amd", "missing_backend"}) {
      bool rejected = false;
      try {
        flagcv::Handle invalid(backend, -1);
      } catch (const flagcv::Error&) {
        rejected = true;
      }
      require(rejected, "invalid backend/device was accepted");
    }
    // HIP image memory remains caller-owned; pitch conversion is in elements.
    void* allocation = nullptr;
    require(hipMalloc(&allocation, 128) == hipSuccess, "hipMalloc failed");
    const flagcv::GpuMat image(allocation, 7, 2, flagcv::Depth::k16U, 3, 64);
    const bool valid = image.row_pitch() == 64 && image.row_stride() == 32;
    require(hipFree(allocation) == hipSuccess, "hipFree failed");
    require(valid, "incorrect 16-bit HIP image stride");
    std::cout << "AMD Handle lifecycle, errors and HIP GpuMat passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
