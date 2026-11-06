/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_BACKENDS_BACKEND_HPP_
#define FLAGCV_BACKENDS_BACKEND_HPP_

#include <cstddef>
#include <cstdint>

/*
 * Private plugin interface: the contract between the core and one platform.
 * Never installed as public API.
 *
 * A plugin does three things: bind a device, turn a compiled kernel into
 * something launchable, and launch it on a stream. Everything above that
 * (validation, kernel choice, caching) is platform-neutral and stays in the
 * core.
 *
 * A plugin must not include a core internal header. It receives the operand
 * layouts and kernel params as plain data, so adding a field to a core struct
 * is not an ABI break for every platform.
 *
 * All callbacks return zero on success and write a diagnostic on failure.
 * No C++ exception or SDK type crosses this boundary. The caller unloads all
 * kernels before destroying their context, and waits for outstanding work
 * before either operation. Contexts are confined to one host thread.
 */

namespace flagcv {

#define FLAGCV_BACKEND_VERSION 1

struct BackendError {
  char message[512] = {};
};

struct LaunchConfig {
  std::uint32_t grid[3];   // Number of blocks, not threads.
  std::uint32_t block[3];  // Threads per block, including the native warp size.
  std::uint32_t shared_memory_bytes;
};

struct BackendApi {
  std::uint32_t version;
  const char* name;
  int (*create_context)(int device, void** context, BackendError*) noexcept;
  int (*destroy_context)(void* context, BackendError*) noexcept;
  int (*device_id)(void* context, char* identity, std::size_t capacity,
                   BackendError*) noexcept;
  // artifact is a path to a fully compiled binary (HSACO on AMD).
  int (*load)(void* context, const char* artifact, const char* entry,
              void** kernel, BackendError*) noexcept;
  int (*unload)(void* context, void* kernel, BackendError*) noexcept;
  // Each argument points to host storage containing one argument's value.
  // stream is caller-owned; nullptr denotes the native default stream.
  int (*launch)(void* context, void* kernel, const LaunchConfig*,
                void** arguments, void* stream, BackendError*) noexcept;
};

using GetBackendApi = const BackendApi* (*)();

}  // namespace flagcv

// Every plugin exports this symbol. Its table remains valid until dlclose().
extern "C" const flagcv::BackendApi* flagcv_backend_api();

#endif  // FLAGCV_BACKENDS_BACKEND_HPP_
