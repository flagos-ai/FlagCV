/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_BACKENDS_BACKEND_HPP_
#define FLAGCV_BACKENDS_BACKEND_HPP_

#include <string>

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
 * TODO: declare the entry table:
 *   create_context(device) / destroy_context()
 *   device_id()        identifies the device for the cache key
 *   load(artifact)     compiled bytes -> launchable
 *   launch(..., stream)  must not compile, autotune or synchronize
 */

namespace flagcv {

#define FLAGCV_BACKEND_VERSION 1

}  // namespace flagcv

#endif  // FLAGCV_BACKENDS_BACKEND_HPP_
