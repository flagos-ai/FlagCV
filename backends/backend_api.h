/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_BACKENDS_BACKEND_API_H_
#define FLAGCV_BACKENDS_BACKEND_API_H_

/*
 * Private versioned plugin ABI. This header is never installed as public API:
 * it is the contract between the core runtime and a backend plugin, and it is
 * free to change between releases as long as the version below moves with it.
 *
 * TODO: declare the V1 entry table.
 *
 * Expected shape, per plugin:
 *   create_context(device_ordinal) / destroy_context()
 *   get_target_fingerprint()        identifies the device for the cache key
 *   create_executable(build_input) / destroy_executable()
 *   execute(executable, stream, bindings, workspace)
 *
 * execute() must not compile, autotune or synchronize. Plugins are discovered
 * by name and loaded with dlopen by src/backend_loader.
 */

#define FLAGCV_BACKEND_API_VERSION 1U

#endif  /* FLAGCV_BACKENDS_BACKEND_API_H_ */
