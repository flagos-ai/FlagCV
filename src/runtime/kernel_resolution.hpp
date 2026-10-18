/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_RUNTIME_KERNEL_RESOLUTION_HPP_
#define FLAGCV_SRC_RUNTIME_KERNEL_RESOLUTION_HPP_

#include <string>
#include <string_view>
#include <vector>

namespace flagcv {

enum class RegistryOwnership {
  /* backends/<platform>/kernels/registry.json */
  kPlatform,
  /* kernels/registry.json */
  kCommon,
};

struct ResolvedKernel {
  RegistryOwnership ownership = RegistryOwnership::kCommon;
  /* Path relative to the owning registry's kernel directory. */
  std::string source;
  std::vector<std::string> functions;
  std::string provider;
  /* Empty when the registry entry declares no tuning section. */
  std::string tuning_source;
  std::string tuning_table;
};

/*
 * Resolves an operation to a kernel using the two-tier registry.
 *
 * Lookup order is the platform registry first, then the common registry. The
 * platform tier is exclusive: once an operation is registered there, a later
 * failure is reported rather than silently retried against the common kernel.
 * Any layout constraint a platform kernel relies on therefore has to be
 * checked by that platform's dispatch layer at build time.
 *
 * TODO: implement. Reads the registries installed under
 * <prefix>/share/flagcv/kernels.
 */
ResolvedKernel resolve_kernel(std::string_view backend_name,
                              std::string_view kernel_operation);

}  // namespace flagcv

#endif  // FLAGCV_SRC_RUNTIME_KERNEL_RESOLUTION_HPP_
