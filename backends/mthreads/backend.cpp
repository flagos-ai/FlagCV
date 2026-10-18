/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/backend.hpp"

/*
 * MThreads plugin.
 *
 * TODO: implement the entry table declared in backends/backend.hpp.
 *
 * Binds a device, loads compiled kernels and launches them. `Stream` is
 * reinterpreted as musaStream_t here. launch() must not compile, autotune or
 * synchronize.
 *
 * Include no core internal header: this plugin receives operand layouts and
 * kernel params as plain data.
 */
