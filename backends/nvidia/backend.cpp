/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/backend.hpp"

#include <cuda.h>

#include <cstdio>
#include <string>

/*
 * NVIDIA plugin.
 *
 * Binds a device through its primary context, the same one the CUDA runtime
 * (and therefore Torch or OpenCV CUDA) uses, so device pointers and streams
 * from those libraries are valid here. Every entry pushes the context for its
 * own duration and pops it, leaving the caller's current context untouched.
 *
 * Includes no core internal header: it receives compiled bytes, launch
 * geometry and argument pointers as plain data.
 */

namespace {

thread_local std::string last_error_message;

int fail(const char* what, CUresult result) {
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &text);
  last_error_message = std::string("nvidia: ") + what +
                       " failed: " + (name ? name : "?") + " (" +
                       (text ? text : "?") + ")";
  return 1;
}

int fail(const std::string& message) {
  last_error_message = "nvidia: " + message;
  return 1;
}

struct Context {
  int device = 0;
  CUdevice handle = 0;
  CUcontext context = nullptr;
};

struct Kernel {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
};

class Scope {
 public:
  explicit Scope(CUcontext context)
      : ok_(cuCtxPushCurrent(context) == CUDA_SUCCESS) {}
  ~Scope() {
    if (ok_) {
      CUcontext popped;
      cuCtxPopCurrent(&popped);
    }
  }
  bool ok() const { return ok_; }

 private:
  bool ok_;
};

int write_string(const std::string& value, char* buffer, size_t size) {
  if (value.size() + 1 > size) return fail("output buffer too small");
  std::snprintf(buffer, size, "%s", value.c_str());
  return 0;
}

int create_context(int device, void** out) {
  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) return fail("cuInit", r);
  auto* ctx = new Context;
  ctx->device = device;
  if ((r = cuDeviceGet(&ctx->handle, device)) != CUDA_SUCCESS) {
    delete ctx;
    return fail("cuDeviceGet", r);
  }
  if ((r = cuDevicePrimaryCtxRetain(&ctx->context, ctx->handle)) !=
      CUDA_SUCCESS) {
    delete ctx;
    return fail("cuDevicePrimaryCtxRetain", r);
  }
  *out = ctx;
  return 0;
}

void destroy_context(void* context) {
  auto* ctx = static_cast<Context*>(context);
  if (ctx == nullptr) return;
  cuDevicePrimaryCtxRelease(ctx->handle);
  delete ctx;
}

int compute_capability(const Context* ctx, int* major, int* minor) {
  CUresult r = cuDeviceGetAttribute(
      major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, ctx->handle);
  if (r != CUDA_SUCCESS) return fail("cuDeviceGetAttribute(major)", r);
  r = cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           ctx->handle);
  if (r != CUDA_SUCCESS) return fail("cuDeviceGetAttribute(minor)", r);
  return 0;
}

int device_id(void* context, char* buffer, size_t size) {
  auto* ctx = static_cast<Context*>(context);
  CUuuid uuid;
  CUresult r = cuDeviceGetUuid(&uuid, ctx->handle);
  if (r != CUDA_SUCCESS) return fail("cuDeviceGetUuid", r);
  int major = 0, minor = 0;
  if (compute_capability(ctx, &major, &minor)) return 1;
  std::string id = "cuda:sm" + std::to_string(major * 10 + minor) + ":";
  char hex[3];
  for (unsigned char byte : uuid.bytes) {
    std::snprintf(hex, sizeof(hex), "%02x", byte);
    id += hex;
  }
  return write_string(id, buffer, size);
}

int target(void* context, char* buffer, size_t size) {
  auto* ctx = static_cast<Context*>(context);
  int major = 0, minor = 0, warp = 0;
  if (compute_capability(ctx, &major, &minor)) return 1;
  CUresult r =
      cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, ctx->handle);
  if (r != CUDA_SUCCESS) return fail("cuDeviceGetAttribute(warp size)", r);
  return write_string(
      "cuda:" + std::to_string(major * 10 + minor) + ":" + std::to_string(warp),
      buffer, size);
}

int load(void* context, const void* binary, size_t, const char* function,
         unsigned int shared_bytes, void** out) {
  auto* ctx = static_cast<Context*>(context);
  Scope scope(ctx->context);
  if (!scope.ok()) return fail("cannot make the device context current");

  auto* kernel = new Kernel;
  CUresult r = cuModuleLoadData(&kernel->module, binary);
  if (r != CUDA_SUCCESS) {
    delete kernel;
    return fail("cuModuleLoadData", r);
  }
  if ((r = cuModuleGetFunction(&kernel->function, kernel->module, function)) !=
      CUDA_SUCCESS) {
    cuModuleUnload(kernel->module);
    delete kernel;
    return fail("cuModuleGetFunction", r);
  }
  // Above 48 KiB a kernel must opt in to dynamic shared memory explicitly.
  if (shared_bytes > 48 * 1024) {
    int limit = 0;
    cuDeviceGetAttribute(&limit,
                         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                         ctx->handle);
    if (static_cast<int>(shared_bytes) > limit ||
        (r = cuFuncSetAttribute(
             kernel->function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
             static_cast<int>(shared_bytes))) != CUDA_SUCCESS) {
      cuModuleUnload(kernel->module);
      delete kernel;
      return r == CUDA_SUCCESS
                 ? fail("kernel needs " + std::to_string(shared_bytes) +
                        " bytes of shared memory, device allows " +
                        std::to_string(limit))
                 : fail("cuFuncSetAttribute", r);
    }
  }
  *out = kernel;
  return 0;
}

void unload(void* context, void* kernel_ptr) {
  auto* ctx = static_cast<Context*>(context);
  auto* kernel = static_cast<Kernel*>(kernel_ptr);
  if (kernel == nullptr) return;
  Scope scope(ctx->context);
  cuModuleUnload(kernel->module);
  delete kernel;
}

int launch(void* context, void* kernel_ptr, const unsigned int grid[3],
           unsigned int block_threads, unsigned int shared_bytes, void* stream,
           void** args) {
  auto* ctx = static_cast<Context*>(context);
  auto* kernel = static_cast<Kernel*>(kernel_ptr);
  if (grid[0] == 0 || grid[1] == 0 || grid[2] == 0) return 0;
  Scope scope(ctx->context);
  if (!scope.ok()) return fail("cannot make the device context current");
  CUresult r = cuLaunchKernel(kernel->function, grid[0], grid[1], grid[2],
                              block_threads, 1, 1, shared_bytes,
                              static_cast<CUstream>(stream), args, nullptr);
  if (r != CUDA_SUCCESS) return fail("cuLaunchKernel", r);
  return 0;
}

const char* last_error() { return last_error_message.c_str(); }

}  // namespace

extern "C" __attribute__((visibility("default"))) const FlagcvBackendV1*
flagcv_backend_v1() {
  static const FlagcvBackendV1 table = {
      FLAGCV_BACKEND_VERSION,
      create_context,
      destroy_context,
      device_id,
      target,
      load,
      unload,
      launch,
      last_error,
  };
  return &table;
}
