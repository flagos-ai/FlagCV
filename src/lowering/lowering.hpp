/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_LOWERING_LOWERING_HPP_
#define FLAGCV_SRC_LOWERING_LOWERING_HPP_

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
  /* Registry key naming the kernel to resolve. */
  std::string kernel;

  /* Image operands in kernel argument order. */
  std::vector<ImageSpec> images;

  /* Integer kernel arguments. Row strides here are in elements, not bytes. */
  std::vector<std::pair<std::string, std::int64_t>> params;

  void add_image(const ImageSpec& image) { images.push_back(image); }

  void add_param(std::string name, std::int64_t value) {
    params.emplace_back(std::move(name), value);
  }
};

/*
 * One lowering function per operator family, defined in
 * src/lowering/<family>.cpp and declared here. None implemented yet.
 *
 * There is no central dispatch: an entry point knows its own family and calls
 * its lowering directly.
 */

}  // namespace flagcv

#endif  // FLAGCV_SRC_LOWERING_LOWERING_HPP_
