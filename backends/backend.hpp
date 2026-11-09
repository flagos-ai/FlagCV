/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_BACKENDS_BACKEND_HPP_
#define FLAGCV_BACKENDS_BACKEND_HPP_

#include <stddef.h>

/*
 * Private plugin interface: the contract between the core and one platform.
 * Never installed as public API.
 *
 * A plugin does three things: bind a device, turn a compiled kernel into
 * something launchable, and launch it on a stream. Everything above that
 * (validation, kernel choice, caching, compiling) is platform-neutral and
 * stays in the core or the external compiler.
 *
 * A plugin must not include a core internal header. It receives compiled
 * bytes, launch geometry and an argument array as plain data, so adding a
 * field to a core struct is not an ABI break for every platform. The table is
 * plain C so a plugin built by another compiler still loads.
 *
 * Every int-returning entry returns 0 on success; on failure it returns
 * non-zero and last_error() describes the failure on the calling thread.
 */

#define FLAGCV_BACKEND_VERSION 2
#define FLAGCV_BACKEND_ENTRY "flagcv_backend_v2"

/* load() flags. */
/* The binary is runtime-registered (Ascend .npubin) rather than a module. */
#define FLAGCV_LOAD_RUNTIME_REGISTER 1u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FlagcvBackendV2 {
  /* FLAGCV_BACKEND_VERSION the plugin was built against. */
  int version;

  /*
   * 1 when compiled kernels take Triton's two trailing scratch pointers
   * after the declared arguments, 0 when the platform's ABI does not (the
   * Ascend path does not). The core sizes the argument array from this.
   */
  int appends_scratch_args;

  /* Bind `device` and return an opaque context for the calls below. */
  int (*create_context)(int device, void** context);
  void (*destroy_context)(void* context);

  /* Identifies the physical device for the kernel cache key. */
  int (*device_id)(void* context, char* buffer, size_t size);

  /*
   * Compile target in the external compiler's "<backend>:<arch>:<warp size>"
   * form, e.g. "cuda:90:32" or "hip:gfx942:64". Supplying it here is what
   * lets the compiler process build for the device without initialising one
   * itself.
   */
  int (*target)(void* context, char* buffer, size_t size);

  /* Compiled bytes -> launchable kernel named `function`. */
  int (*load)(void* context, const void* binary, size_t size,
              const char* function, unsigned int shared_bytes,
              unsigned int flags, void** kernel);
  void (*unload)(void* context, void* kernel);

  /*
   * Enqueue on `stream` (null = default stream). `args` follows the native
   * launch convention: one pointer per kernel argument, pointing at its value,
   * in kernel order and including the trailing scratch pointers when
   * appends_scratch_args is set. `kinds` has one entry per element of `args`:
   * 0 for a 64-bit value (a device pointer or an i64), 1 for a 32-bit value
   * (an i32). Backends that launch through a pointer array ignore `kinds`;
   * Ascend packs arguments by width and needs it.
   * Must not compile, autotune or synchronize.
   */
  int (*launch)(void* context, void* kernel, const unsigned int grid[3],
                unsigned int block_threads, unsigned int shared_bytes,
                void* stream, void** args, const unsigned int* kinds,
                unsigned int num_args);

  const char* (*last_error)(void);
} FlagcvBackendV2;

/* Exported by every plugin under the name FLAGCV_BACKEND_ENTRY. */
typedef const FlagcvBackendV2* (*FlagcvBackendEntry)(void);

#ifdef __cplusplus
}
#endif

#endif  // FLAGCV_BACKENDS_BACKEND_HPP_
