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
 * Signature convention:
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

/*
 * Mirrors cv::cuda::flip(src, dst, flipCode, stream).
 *
 * flip_code == 0 flips around the x-axis (rows reversed), > 0 around the
 * y-axis (columns reversed), < 0 around both. Every depth and 1-4 channels
 * are supported; pixels move bit-for-bit. `dst` must match `src` in extent,
 * depth and channels, and must not overlap it: in-place flipping is not
 * supported. Pitches may differ.
 */
void flip(Handle& handle,
          const GpuMat& src,
          const GpuMat& dst,
          int flip_code,
          Stream stream = nullptr);

}  // namespace flagcv

#endif  // FLAGCV_CORE_HPP_
