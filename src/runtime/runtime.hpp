/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_RUNTIME_RUNTIME_HPP_
#define FLAGCV_SRC_RUNTIME_RUNTIME_HPP_

#include <flagcv/handle.hpp>
#include <flagcv/types.hpp>

#include "src/lowering/lowering.hpp"

#include <vector>

namespace flagcv {

/*
 * The single execution path every operator entry point funnels into.
 *
 * Resolves the kernel, compiles on a cache miss, launches on `stream`. Does
 * not synchronize.
 *
 * Lowering happens in the entry point, not here, so invalid operands are
 * rejected on every call rather than only on a cache miss.
 */
void launch(Handle& handle,
            const LoweredOp& op,
            const std::vector<void*>& pointers,
            Stream stream);

/*
 * Resolves an operation to a kernel source and entry function.
 *
 * Looks in backends/<backend>/kernels/registry.json first, then
 * kernels/registry.json. The platform tier is EXCLUSIVE: once an operation is
 * registered there, a failure is reported rather than silently retried
 * against the common kernel, so any layout constraint a platform kernel needs
 * must be checked by that platform's dispatch layer before compilation.
 */
struct ResolvedKernel {
  std::string source;
  std::string function;
  bool from_platform_registry = false;
};

ResolvedKernel resolve_kernel(const std::string& backend,
                              const std::string& kernel);

}  // namespace flagcv

#endif  // FLAGCV_SRC_RUNTIME_RUNTIME_HPP_
