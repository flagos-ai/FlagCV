/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_CORE_HPP_
#define FLAGCV_CORE_HPP_

#include <flagcv/gpumat.hpp>
#include <flagcv/handle.hpp>

/*
 * Per-element arithmetic and logic, comparison, reduction, channel handling.
 * Mirrors opencv2/core.hpp, so a declaration added here keeps the OpenCV name
 * and argument order.
 *
 * No operators yet. Signature convention:
 *
 *   void add(Handle& handle,
 *            const GpuMat& src1,
 *            const GpuMat& src2,
 *            const GpuMat& dst,
 *            const GpuMat& mask = GpuMat(),
 *            Stream stream = nullptr);
 *
 * `handle` leads because it carries the backend and the kernel cache;
 * `stream` trails with a default so the common call reads like OpenCV. Calls
 * enqueue and do not synchronize. Operand validation belongs in
 * src/lowering/, never in the entry point. See docs/operator-development.md.
 */
namespace flagcv {

/* Operator declarations go here. */

}  // namespace flagcv

#endif  // FLAGCV_CORE_HPP_
