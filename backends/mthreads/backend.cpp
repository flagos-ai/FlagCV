/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <musa.h>

#include "backends/common/driver_backend.hpp"

/*
 * MThreads (MUSA) plugin.
 *
 * The MUSA driver API mirrors the CUDA one, so the shared flow applies. The
 * compile target uses Triton's "mtgpu" backend name (the same mapping
 * libtriton_jit applies to MUSA) and the artifact is a .mubin.
 *
 * Includes no core internal header.
 */

namespace {

struct MusaApi {
  using Status = MUresult;
  using Device = MUdevice;
  using Context = MUcontext;
  using Module = MUmodule;
  using Function = MUfunction;

  static const char* name() { return "mthreads"; }
  static Status success() { return MUSA_SUCCESS; }

  static Status init() { return muInit(0); }
  static Status device_get(Device* device, int index) {
    return muDeviceGet(device, index);
  }
  static Status primary_ctx_retain(Context* ctx, Device device) {
    return muDevicePrimaryCtxRetain(ctx, device);
  }
  static Status primary_ctx_release(Device device) {
    return muDevicePrimaryCtxRelease(device);
  }
  static Status ctx_push(Context ctx) { return muCtxPushCurrent(ctx); }
  static Status ctx_pop(Context* ctx) { return muCtxPopCurrent(ctx); }
  static Status module_load_data(Module* module, const void* data) {
    return muModuleLoadData(module, data);
  }
  static Status module_get_function(Function* fn, Module module,
                                    const char* name) {
    return muModuleGetFunction(fn, module, name);
  }
  static Status module_unload(Module module) { return muModuleUnload(module); }
  static Status launch_kernel(Function fn, unsigned gx, unsigned gy,
                              unsigned gz, unsigned block, unsigned shared,
                              void* stream, void** args) {
    return muLaunchKernel(fn, gx, gy, gz, block, 1, 1, shared,
                          static_cast<MUstream>(stream), args, nullptr);
  }
  static Status set_shared(Function fn, int bytes) {
    return muFuncSetAttribute(
        fn, MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
  }

  static Status attribute(int* value, MUdevice_attribute what, Device device) {
    return muDeviceGetAttribute(value, what, device);
  }

  static Status target(Device device, std::string& out) {
    int major = 0, minor = 0, warp = 0;
    if (attribute(&major, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                  device) != success() ||
        attribute(&minor, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                  device) != success() ||
        attribute(&warp, MU_DEVICE_ATTRIBUTE_WARP_SIZE, device) != success()) {
      return MUSA_ERROR_UNKNOWN;
    }
    out = "mtgpu:" + std::to_string(major * 10 + minor) + ":" +
          std::to_string(warp);
    return MUSA_SUCCESS;
  }

  static Status device_key(Device device, std::string& out) {
    int major = 0, minor = 0, bus = 0;
    if (attribute(&major, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                  device) != success() ||
        attribute(&minor, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                  device) != success() ||
        attribute(&bus, MU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device) != success()) {
      return MUSA_ERROR_UNKNOWN;
    }
    out = "mtgpu:sm" + std::to_string(major * 10 + minor) + ":bus" +
          std::to_string(bus);
    return MUSA_SUCCESS;
  }

  static int shared_optin_limit(Device device) {
    int limit = 0;
    muDeviceGetAttribute(
        &limit, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);
    return limit;
  }
  static unsigned shared_optin_threshold() { return 48u * 1024u; }
};

}  // namespace

extern "C" __attribute__((visibility("default"))) const FlagcvBackendV2*
flagcv_backend_v2() {
  return flagcv::driver::table<MusaApi>();
}
