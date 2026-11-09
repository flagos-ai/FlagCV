/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_LOWERING_LOWERING_HPP_
#define FLAGCV_SRC_LOWERING_LOWERING_HPP_

#include <flagcv/gpumat.hpp>

#include "src/image/types.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace flagcv {

/*
 * Result of lowering one operator call: validated operands plus the kernel's
 * integer arguments.
 *
 * Lowering checks public semantics only. Block sizes, shared memory and grid
 * extents are platform properties decided by the platform compiler, so adding
 * a platform never touches a lowering file.
 *
 * Why this layer exists at all in an immediate-mode runtime: several entry
 * points share one family (bitwise_and/or/xor/not would be four entries over
 * one family), and keeping validation here rather than in the entry points is
 * what stops them from drifting on what operands they accept.
 */
struct LoweredOp {
  /*
   * One integer kernel argument. A compile-time param becomes a Triton
   * constexpr: its value enters the kernel cache key and each new value
   * compiles a new kernel. A runtime param is passed at launch: only its type
   * enters the key, so new values (image sizes, strides) reuse the kernel.
   * Keep compile-time params to small closed sets such as modes and channel
   * counts.
   */
  struct Param {
    std::string name;
    std::int64_t value = 0;
    bool compile_time = false;
    /*
     * Runtime params only: also specialise on value % 16 == 0, the way
     * Triton's JIT does. Worth it where the compiler needs that fact to
     * vectorise: strides (row starts provably aligned) and the extent that
     * bounds the vectorised axis (the mask is then constant over 16-element
     * groups). Not for other extents, where it only doubles the variants.
     */
    bool divisibility_hint = false;
  };

  /* Registry key naming the kernel to resolve. */
  std::string kernel;

  /* Image operands in kernel argument order. */
  std::vector<ImageSpec> images;

  /*
   * Integer kernel arguments, in kernel argument order after the image
   * pointers. Row strides here are in elements, not bytes.
   */
  std::vector<Param> params;

  void add_image(const ImageSpec& image) { images.push_back(image); }

  void add_param(std::string name, std::int64_t value) {
    params.push_back(Param{std::move(name), value, false, false});
  }

  /* A runtime param that also specialises on % 16; see divisibility_hint. */
  void add_hinted_param(std::string name, std::int64_t value) {
    params.push_back(Param{std::move(name), value, false, true});
  }

  void add_constexpr(std::string name, std::int64_t value) {
    params.push_back(Param{std::move(name), value, true, false});
  }
};

/*
 * One lowering function per operator family, defined in
 * src/lowering/<family>.cpp and declared here.
 *
 * There is no central dispatch: an entry point knows its own family and calls
 * its lowering directly.
 */

/* flip: flip_code == 0 mirrors rows, > 0 mirrors columns, < 0 both. */
LoweredOp lower_flip(const GpuMat& src, const GpuMat& dst, int flip_code);

}  // namespace flagcv

#endif  // FLAGCV_SRC_LOWERING_LOWERING_HPP_
