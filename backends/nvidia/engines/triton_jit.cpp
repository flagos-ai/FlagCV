/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

/*
 * TODO: Triton JIT execution adapter.
 *
 * Packed kernels take int32-typed pointers and word-counted row strides, so
 * this layer converts the byte pitches in ImageSpec into word strides before
 * launching. Mixing the two unit systems is the main hazard here.
 */

#include "backends/nvidia/engines/engine.hpp"
