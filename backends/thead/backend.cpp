/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <hggc.h>

#include "backends/common/driver_backend.hpp"

/*
 * T-Head PPU plugin.
 *
 * The HGGC driver API follows the CUDA conventions (libtriton_jit's PPU
 * backend says the same), so the shared flow applies with the hg* names, a
 * 32-lane warp and the .hgbin artifact. The compile target uses Triton's
 * "PPU" backend name.
 *
 * Includes no core internal header.
 */

namespace {

struct PpuApi {
  using Status = HGresult;
  using Device = HGdevice;
  using Context = HGcontext;
  using Module = HGmodule;
  using Function = HGfunction;

  static const char* name() { return "thead"; }
  static Status success() { return HGGC_SUCCESS; }

  static Status init() { return hgInit(0); }
  static Status device_get(Device* device, int index) {
    return hgDeviceGet(device, index);
  }
  static Status primary_ctx_retain(Context* ctx, Device device) {
    return hgDevicePrimaryCtxRetain(ctx, device);
  }
  static Status primary_ctx_release(Device device) {
    return hgDevicePrimaryCtxRelease(device);
  }
  static Status ctx_push(Context ctx) { return hgCtxPushCurrent(ctx); }
  static Status ctx_pop(Context* ctx) { return hgCtxPopCurrent(ctx); }
  static Status module_load_data(Module* module, const void* data) {
    return hgModuleLoadData(module, data);
  }
  static Status module_get_function(Function* fn, Module module,
                                    const char* name) {
    return hgModuleGetFunction(fn, module, name);
  }
  static Status module_unload(Module module) { return hgModuleUnload(module); }
  static Status launch_kernel(Function fn, unsigned gx, unsigned gy,
                              unsigned gz, unsigned block, unsigned shared,
                              void* stream, void** args) {
    return hgLaunchKernel(fn, gx, gy, gz, block, 1, 1, shared,
                          static_cast<HGstream>(stream), args, nullptr);
  }
  static Status set_shared(Function fn, int bytes) {
    return hgFuncSetAttribute(
        fn, HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
  }

  static Status attribute(int* value, HGdevice_attribute what, Device device) {
    return hgDeviceGetAttribute(value, what, device);
  }

  static Status target(Device device, std::string& out) {
    int major = 0, minor = 0, warp = 0;
    if (attribute(&major, HG_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                  device) != success() ||
        attribute(&minor, HG_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                  device) != success() ||
        attribute(&warp, HG_DEVICE_ATTRIBUTE_WARP_SIZE, device) != success()) {
      return HGGC_ERROR_UNKNOWN;
    }
    out = "PPU:" + std::to_string(major * 10 + minor) + ":" +
          std::to_string(warp);
    return HGGC_SUCCESS;
  }

  static Status device_key(Device device, std::string& out) {
    int major = 0, minor = 0, bus = 0;
    if (attribute(&major, HG_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                  device) != success() ||
        attribute(&minor, HG_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                  device) != success() ||
        attribute(&bus, HG_DEVICE_ATTRIBUTE_PCI_BUS_ID, device) != success()) {
      return HGGC_ERROR_UNKNOWN;
    }
    out = "PPU:sm" + std::to_string(major * 10 + minor) + ":bus" +
          std::to_string(bus);
    return HGGC_SUCCESS;
  }

  static int shared_optin_limit(Device device) {
    int limit = 0;
    hgDeviceGetAttribute(
        &limit, HG_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);
    return limit;
  }
  static unsigned shared_optin_threshold() { return 48u * 1024u; }
};

}  // namespace

extern "C" __attribute__((visibility("default"))) const FlagcvBackendV2*
flagcv_backend_v2() {
  return flagcv::driver::table<PpuApi>();
}
