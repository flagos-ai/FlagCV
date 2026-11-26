/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/backend.hpp"

#include <acl/acl.h>
#include <runtime/runtime/rt.h>

#include <cstring>
#include <string>
#include <vector>

/*
 * Ascend plugin.
 *
 * Ascend does not load modules the CUDA way: the compiled binary is
 * registered with the runtime (rtDevBinaryRegister), a stub is registered for
 * the kernel entry point (rtFunctionRegister), and launches go through
 * rtKernelLaunch with a *packed* argument buffer rather than an array of
 * argument pointers. So this plugin implements the table itself instead of
 * using backends/common/driver_backend.hpp.
 *
 * The packed layout follows libtriton_jit's Ascend backend: three 8-byte
 * system arguments first (ffts address, block-lock pointer, workspace
 * address), then the user arguments at their natural alignment, then the
 * launch grid as three int32 values. Triton's Ascend ABI does not append the
 * scratch pointers, so appends_scratch_args is 0.
 *
 * Includes no core internal header.
 */

namespace {

std::string& error_slot() {
  static thread_local std::string message;
  return message;
}

int fail(const std::string& what) {
  error_slot() = "ascend: " + what;
  return 1;
}

/* One 8-byte system argument, one pointer, or one 32/64-bit integer. */
struct ArgBuffer {
  std::vector<unsigned char> bytes;

  void align_to(size_t alignment) {
    while (bytes.size() % alignment != 0) bytes.push_back(0);
  }

  void append(const void* value, size_t size, size_t alignment) {
    align_to(alignment);
    const auto* source = static_cast<const unsigned char*>(value);
    bytes.insert(bytes.end(), source, source + size);
  }
};

size_t align_of(unsigned int kind) {
  return kind == 1 ? sizeof(int32_t) : sizeof(int64_t);
}
size_t size_of(unsigned int kind) {
  return kind == 1 ? sizeof(int32_t) : sizeof(int64_t);
}

struct Context {
  int index = 0;
  aclrtStream stream = nullptr;
};

struct Kernel {
  rtDevBinary_t binary{};
  std::vector<unsigned char>
      image;  // owns the bytes rtDevBinaryRegister points at
  void* bin_handle = nullptr;
  std::vector<unsigned char*> stubs;  // registered entry-point handles
  void* stub = nullptr;
  std::string name;
};

int create_context(int device, void** out) {
  if (aclInit(nullptr) != ACL_ERROR_NONE) return fail("aclInit failed");
  if (aclrtSetDevice(device) != ACL_ERROR_NONE)
    return fail("aclrtSetDevice failed");
  auto* ctx = new Context;
  ctx->index = device;
  *out = ctx;
  return 0;
}

void destroy_context(void* context) {
  auto* ctx = static_cast<Context*>(context);
  if (ctx == nullptr) return;
  aclrtResetDevice(ctx->index);
  aclFinalize();
  delete ctx;
}

int device_id(void* context, char* buffer, size_t size) {
  auto* ctx = static_cast<Context*>(context);
  // Ascend exposes chip identity through the SocName and the device index;
  // together they are stable for a given physical device on a host.
  const char* soc = aclrtGetSocName();
  const std::string key = std::string("ascend:") + (soc ? soc : "unknown") +
                          ":" + std::to_string(ctx->index);
  if (key.size() + 1 > size) return fail("output buffer too small");
  std::snprintf(buffer, size, "%s", key.c_str());
  return 0;
}

int target(void* context, char* buffer, size_t size) {
  (void)context;
  // Triton's Ascend backend names its target "npu"; warp size is 1 because
  // the platform does not use SIMT warps.
  const char* spec = "npu:0:1";
  if (std::strlen(spec) + 1 > size) return fail("output buffer too small");
  std::snprintf(buffer, size, "%s", spec);
  return 0;
}

int load(void* context, const void* binary, size_t size, const char* function,
         unsigned int, unsigned int flags, void** out) {
  (void)context;
  if ((flags & FLAGCV_LOAD_RUNTIME_REGISTER) == 0) {
    return fail(
        "Ascend binaries must be loaded with FLAGCV_LOAD_RUNTIME_REGISTER");
  }
  auto* kernel = new Kernel;
  kernel->name = function;
  kernel->image.assign(static_cast<const unsigned char*>(binary),
                       static_cast<const unsigned char*>(binary) + size);
  kernel->binary.data = kernel->image.data();
  kernel->binary.length = static_cast<uint32_t>(size);
  // AIVECTOR is the vector-core image Triton emits for Ascend; the plain ELF
  // magic is the fallback used by libtriton_jit for non-"aiv" kernels.
  kernel->binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  kernel->binary.version = 0;
  if (rtDevBinaryRegister(&kernel->binary, &kernel->bin_handle) !=
      RT_ERROR_NONE) {
    delete kernel;
    return fail("rtDevBinaryRegister failed");
  }
  kernel->stubs.push_back(nullptr);
  void* stub = kernel->stubs.back();
  if (rtFunctionRegister(kernel->bin_handle, &stub, function,
                         const_cast<char*>(function), 0) != RT_ERROR_NONE) {
    delete kernel;
    return fail("rtFunctionRegister failed");
  }
  kernel->stubs.back() = static_cast<unsigned char*>(stub);
  kernel->stub = stub;
  *out = kernel;
  return 0;
}

void unload(void* context, void* kernel_ptr) {
  (void)context;
  auto* kernel = static_cast<Kernel*>(kernel_ptr);
  if (kernel == nullptr) return;
  rtDevBinaryUnRegister(kernel->bin_handle);
  delete kernel;
}

int launch(void* context, void* kernel_ptr, const unsigned int grid[3],
           unsigned int block_threads, unsigned int, void* stream, void** args,
           const unsigned int* kinds, unsigned int num_args) {
  auto* ctx = static_cast<Context*>(context);
  auto* kernel = static_cast<Kernel*>(kernel_ptr);
  const uint32_t blocks = grid[0] * grid[1] * grid[2];
  if (blocks == 0) return 0;

  void* ffts = nullptr;
  uint32_t ffts_len = 0;
  if (rtGetC2cCtrlAddr(reinterpret_cast<uint64_t*>(&ffts), &ffts_len) !=
      RT_ERROR_NONE) {
    return fail("rtGetC2cCtrlAddr failed");
  }

  ArgBuffer packed;
  void* system[3] = {ffts, nullptr, nullptr};  // ffts, block lock, workspace
  for (void* value : system)
    packed.append(&value, sizeof(void*), alignof(void*));
  for (unsigned int i = 0; i < num_args; ++i) {
    packed.append(args[i], size_of(kinds[i]), align_of(kinds[i]));
  }
  for (int axis = 0; axis < 3; ++axis) {
    const int32_t extent = static_cast<int32_t>(grid[axis]);
    packed.append(&extent, sizeof(int32_t), alignof(int32_t));
  }

  const rtError_t result =
      rtKernelLaunch(kernel->stub, blocks, packed.bytes.data(),
                     static_cast<uint32_t>(packed.bytes.size()), nullptr,
                     static_cast<aclrtStream>(stream ? stream : ctx->stream));
  if (result != RT_ERROR_NONE)
    return fail("rtKernelLaunch failed (block threads " +
                std::to_string(block_threads) + ")");
  return 0;
}

const char* last_error() { return error_slot().c_str(); }

}  // namespace

extern "C" __attribute__((visibility("default"))) const FlagcvBackendV2*
flagcv_backend_v2() {
  static const FlagcvBackendV2 table = {
      FLAGCV_BACKEND_VERSION,
      0,  // appends_scratch_args
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
