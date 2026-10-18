/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/runtime/runtime.hpp"

namespace flagcv {

/*
 * TODO: implement launch().
 *
 *   1. Build a cache key from the kernel name, every operand's layout, every
 *      param, the device and the compiler identity. Pixel values never enter
 *      the key.
 *   2. On a hit, go to step 4.
 *   3. On a miss, resolve the kernel, hand the lowered form to the external
 *      compiler (compiler/flagcv_codegen), cache the result.
 *   4. Launch on the caller's stream. No compiling, no autotuning and no
 *      synchronizing on this step.
 */
void launch(Handle& handle,
            const LoweredOp& op,
            const std::vector<void*>& pointers,
            Stream stream) {
  (void)handle;
  (void)op;
  (void)pointers;
  (void)stream;
  throw Error("flagcv: execution is not implemented yet");
}

/*
 * TODO: implement resolve_kernel().
 *
 * Read the platform registry, then the common one. If neither registers the
 * operation, throw. Do not treat a platform entry whose constraints fail as a
 * miss: that would make platform registration non-exclusive and hide a
 * dispatch-layer bug.
 */
ResolvedKernel resolve_kernel(const std::string& backend,
                              const std::string& kernel) {
  (void)backend;
  (void)kernel;
  throw Error("flagcv: kernel resolution is not implemented yet");
}

/*
 * TODO: implement Handle.
 *
 * State holds the loaded backend plugin, the device, and the compiled-kernel
 * cache. Construction loads the plugin by name; destruction releases the
 * cache before the plugin.
 */
struct Handle::State {
  std::string backend;
  int device = 0;
};

Handle::Handle(const std::string& backend, int device)
    : state_(new State{backend, device}) {}

Handle::~Handle() = default;
Handle::Handle(Handle&&) noexcept = default;
Handle& Handle::operator=(Handle&&) noexcept = default;

const std::string& Handle::backend() const {
  return state_->backend;
}

}  // namespace flagcv
