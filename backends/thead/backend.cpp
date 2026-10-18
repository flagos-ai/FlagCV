/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/backend.hpp"

/*
 * T-Head plugin.
 *
 * TODO: implement the entry table declared in backends/backend.hpp.
 *
 * Binds a device, loads compiled kernels and launches them. `Stream` is
 * reinterpreted as CUstream here (the PPU driver API shape). launch() must not
 * compile, autotune or synchronize.
 *
 * Include no core internal header: this plugin receives operand layouts and
 * kernel params as plain data.
 */
