/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_BACKENDS_COMMON_DRIVER_BACKEND_HPP_
#define FLAGCV_BACKENDS_COMMON_DRIVER_BACKEND_HPP_

#include <cstdio>
#include <string>

#include "backends/backend.hpp"

/*
 * The plugin table for any vendor whose driver API has the CUDA shape: init,
 * get device, retain its primary context, push/pop that context, load a
 * module from bytes, get a function from it, launch with an argument array.
 *
 * That covers NVIDIA (cuda.h), Iluvatar (its CUDA-compatible cuda.h), Hygon
 * (hip), MThreads (musa) and T-Head (hggc). Each plugin supplies an `Api`
 * struct naming its own types and calls; the flow below is written once so
 * the five cannot drift apart. A platform whose API does not have that shape
 * (Ascend: runtime binary registration plus a packed argument buffer)
 * implements the table itself.
 *
 * The `Api` contract:
 *
 *   name()                    short name for error messages
 *   Status, success()         result type and its success value
 *   Device, Context, Module, Function
 *   init()                    driver initialisation
 *   device_get(Device*, int)  device index -> handle
 *   primary_ctx_retain/release
 *   ctx_push(Context) / ctx_pop(Context*)
 *   module_load_data(Module*, const void*)
 *   module_get_function(Function*, Module, const char*)
 *   module_unload(Module)
 *   launch_kernel(Function, gx, gy, gz, block, shared, void* stream, void**
 * args) device_key(Device, std::string&)   stable cache identity target(Device,
 * std::string&)       "<triton backend>:<arch>:<warp size>"
 *   shared_optin_limit(Device)         device cap on dynamic shared memory
 *   set_shared(Function, int)          raise the kernel's dynamic shared limit
 *   shared_optin_threshold()           above this, set_shared is required
 */

namespace flagcv {
namespace driver {

inline std::string& error_slot() {
  static thread_local std::string message;
  return message;
}

inline int fail(const std::string& prefix, const std::string& what) {
  error_slot() = prefix + ": " + what;
  return 1;
}

template <typename Api>
struct Context {
  int index = 0;
  typename Api::Device device{};
  typename Api::Context context{};
};

template <typename Api>
struct Kernel {
  typename Api::Module module{};
  typename Api::Function function{};
};

/* Push the context for one call and pop it afterwards, leaving whatever the
 * caller had current untouched. */
template <typename Api>
class Scope {
 public:
  explicit Scope(typename Api::Context context)
      : pushed_(Api::ctx_push(context) == Api::success()) {}
  ~Scope() {
    if (pushed_) {
      typename Api::Context popped;
      Api::ctx_pop(&popped);
    }
  }
  bool ok() const { return pushed_; }

 private:
  bool pushed_;
};

template <typename Api>
int write_string(const std::string& value, char* buffer, size_t size) {
  if (value.size() + 1 > size)
    return fail(Api::name(), "output buffer too small");
  std::snprintf(buffer, size, "%s", value.c_str());
  return 0;
}

template <typename Api>
int create_context(int device, void** out) {
  if (Api::init() != Api::success())
    return fail(Api::name(), "driver initialisation failed");
  auto* ctx = new Context<Api>;
  ctx->index = device;
  if (Api::device_get(&ctx->device, device) != Api::success()) {
    delete ctx;
    return fail(Api::name(), "cannot get device " + std::to_string(device));
  }
  if (Api::primary_ctx_retain(&ctx->context, ctx->device) != Api::success()) {
    delete ctx;
    return fail(Api::name(), "cannot retain the device's primary context");
  }
  *out = ctx;
  return 0;
}

template <typename Api>
void destroy_context(void* context) {
  auto* ctx = static_cast<Context<Api>*>(context);
  if (ctx == nullptr) return;
  Api::primary_ctx_release(ctx->device);
  delete ctx;
}

template <typename Api>
int device_id(void* context, char* buffer, size_t size) {
  auto* ctx = static_cast<Context<Api>*>(context);
  std::string key;
  if (Api::device_key(ctx->device, key) != Api::success()) {
    return fail(Api::name(), "cannot identify the device");
  }
  return write_string<Api>(key, buffer, size);
}

template <typename Api>
int target(void* context, char* buffer, size_t size) {
  auto* ctx = static_cast<Context<Api>*>(context);
  std::string spec;
  if (Api::target(ctx->device, spec) != Api::success()) {
    return fail(Api::name(), "cannot describe the compile target");
  }
  return write_string<Api>(spec, buffer, size);
}

template <typename Api>
int load(void* context, const void* binary, size_t, const char* function,
         unsigned int shared_bytes, unsigned int flags, void** out) {
  if (flags != 0) return fail(Api::name(), "this backend loads modules only");
  auto* ctx = static_cast<Context<Api>*>(context);
  Scope<Api> scope(ctx->context);
  if (!scope.ok())
    return fail(Api::name(), "cannot make the device context current");

  auto* kernel = new Kernel<Api>;
  if (Api::module_load_data(&kernel->module, binary) != Api::success()) {
    delete kernel;
    return fail(Api::name(), "cannot load the compiled module");
  }
  if (Api::module_get_function(&kernel->function, kernel->module, function) !=
      Api::success()) {
    Api::module_unload(kernel->module);
    delete kernel;
    return fail(Api::name(),
                "the module has no kernel named " + std::string(function));
  }
  if (shared_bytes > Api::shared_optin_threshold()) {
    const int limit = Api::shared_optin_limit(ctx->device);
    if (static_cast<int>(shared_bytes) > limit ||
        Api::set_shared(kernel->function, static_cast<int>(shared_bytes)) !=
            Api::success()) {
      Api::module_unload(kernel->module);
      delete kernel;
      return fail(Api::name(),
                  "kernel needs " + std::to_string(shared_bytes) +
                      " bytes of dynamic shared memory, device allows " +
                      std::to_string(limit));
    }
  }
  *out = kernel;
  return 0;
}

template <typename Api>
void unload(void* context, void* kernel_ptr) {
  auto* ctx = static_cast<Context<Api>*>(context);
  auto* kernel = static_cast<Kernel<Api>*>(kernel_ptr);
  if (kernel == nullptr) return;
  Scope<Api> scope(ctx->context);
  Api::module_unload(kernel->module);
  delete kernel;
}

template <typename Api>
int launch(void* context, void* kernel_ptr, const unsigned int grid[3],
           unsigned int block_threads, unsigned int shared_bytes, void* stream,
           void** args, const unsigned int*, unsigned int) {
  auto* ctx = static_cast<Context<Api>*>(context);
  auto* kernel = static_cast<Kernel<Api>*>(kernel_ptr);
  if (grid[0] == 0 || grid[1] == 0 || grid[2] == 0) return 0;
  Scope<Api> scope(ctx->context);
  if (!scope.ok())
    return fail(Api::name(), "cannot make the device context current");
  if (Api::launch_kernel(kernel->function, grid[0], grid[1], grid[2],
                         block_threads, shared_bytes, stream,
                         args) != Api::success()) {
    return fail(Api::name(), "kernel launch failed");
  }
  return 0;
}

template <typename Api>
const char* last_error() {
  return error_slot().c_str();
}

template <typename Api>
const FlagcvBackendV2* table() {
  static const FlagcvBackendV2 value = {
      FLAGCV_BACKEND_VERSION,
      1,  // appends_scratch_args: Triton's ABI ends with two scratch pointers
      create_context<Api>,
      destroy_context<Api>,
      device_id<Api>,
      target<Api>,
      load<Api>,
      unload<Api>,
      launch<Api>,
      last_error<Api>,
  };
  return &value;
}

}  // namespace driver
}  // namespace flagcv

#endif  // FLAGCV_BACKENDS_COMMON_DRIVER_BACKEND_HPP_
