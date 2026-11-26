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
 *            Stream stream = nullptr);
 *
 * `handle` leads because it carries the backend and the kernel cache;
 * `stream` trails with a default so the common call reads like OpenCV. Calls
 * enqueue and do not synchronize. Operand validation belongs in
 * src/lowering/, never in the entry point. See docs/operator-development.md.
 */
namespace flagcv {

/*
 * Array-array profiles matching the Python evaluation operators: arithmetic
 * supports 8U (saturating) and 32F, bitwise supports 8U. All support 1-4
 * channels and independent byte pitches. multiply uses unit scale.
 * Operands must have identical extents, depth and channel count. Exact
 * src/dst aliasing is supported; partial overlap is rejected. Scalar, mask
 * and output-depth conversion overloads are not provided by these profiles.
 */
void add(Handle& handle, const GpuMat& src1, const GpuMat& src2,
         const GpuMat& dst, Stream stream = nullptr);
void subtract(Handle& handle, const GpuMat& src1, const GpuMat& src2,
              const GpuMat& dst, Stream stream = nullptr);
void multiply(Handle& handle, const GpuMat& src1, const GpuMat& src2,
              const GpuMat& dst, Stream stream = nullptr);
void bitwise_and(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                 const GpuMat& dst, Stream stream = nullptr);
void bitwise_not(Handle& handle, const GpuMat& src, const GpuMat& dst,
                 Stream stream = nullptr);

/*
 * Mirrors cv::cuda::flip(src, dst, flipCode, stream).
 *
 * flip_code == 0 flips around the x-axis (rows reversed), > 0 around the
 * y-axis (columns reversed), < 0 around both. Every depth and 1-4 channels
 * are supported; pixels move bit-for-bit. `dst` must match `src` in extent,
 * depth and channels, and must not overlap it: in-place flipping is not
 * supported. Pitches may differ.
 */
void flip(Handle& handle, const GpuMat& src, const GpuMat& dst, int flip_code,
          Stream stream = nullptr);

}  // namespace flagcv

#endif  // FLAGCV_CORE_HPP_
