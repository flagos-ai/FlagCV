/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <cuda.h>

#include "backends/common/driver_backend.hpp"

/*
 * NVIDIA plugin.
 *
 * Binds a device through its primary context, the same one the CUDA runtime
 * (and therefore Torch or OpenCV CUDA) uses, so device pointers and streams
 * from those libraries are valid here.
 *
 * Includes no core internal header: it receives compiled bytes, launch
 * geometry and argument pointers as plain data. The flow itself lives in
 * backends/common/driver_backend.hpp.
 */

namespace {

struct NvidiaApi {
  using Status = CUresult;
  using Device = CUdevice;
  using Context = CUcontext;
  using Module = CUmodule;
  using Function = CUfunction;

  static const char* name() { return "nvidia"; }
  static Status success() { return CUDA_SUCCESS; }

  static Status init() { return cuInit(0); }
  static Status device_get(Device* device, int index) {
    return cuDeviceGet(device, index);
  }
  static Status primary_ctx_retain(Context* ctx, Device device) {
    return cuDevicePrimaryCtxRetain(ctx, device);
  }
  static Status primary_ctx_release(Device device) {
    return cuDevicePrimaryCtxRelease(device);
  }
  static Status ctx_push(Context ctx) { return cuCtxPushCurrent(ctx); }
  static Status ctx_pop(Context* ctx) { return cuCtxPopCurrent(ctx); }
  static Status module_load_data(Module* module, const void* data) {
    return cuModuleLoadData(module, data);
  }
  static Status module_get_function(Function* fn, Module module,
                                    const char* name) {
    return cuModuleGetFunction(fn, module, name);
  }
  static Status module_unload(Module module) { return cuModuleUnload(module); }
  static Status launch_kernel(Function fn, unsigned gx, unsigned gy,
                              unsigned gz, unsigned block, unsigned shared,
                              void* stream, void** args) {
    return cuLaunchKernel(fn, gx, gy, gz, block, 1, 1, shared,
                          static_cast<CUstream>(stream), args, nullptr);
  }
  static Status set_shared(Function fn, int bytes) {
    return cuFuncSetAttribute(
        fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
  }

  static Status attribute(int* value, CUdevice_attribute what, Device device) {
    return cuDeviceGetAttribute(value, what, device);
  }

  /* Compile target: "cuda:<sm><minor>:<warp size>", from the compute
   * capability the driver reports. */
  static Status target(Device device, std::string& out) {
    int major = 0, minor = 0, warp = 0;
    if (attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                  device) != success() ||
        attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                  device) != success() ||
        attribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device) != success()) {
      return CUDA_ERROR_UNKNOWN;
    }
    out = "cuda:" + std::to_string(major * 10 + minor) + ":" +
          std::to_string(warp);
    return CUDA_SUCCESS;
  }

  /* Cache identity: architecture plus PCI bus id, which is stable for a given
   * physical device even when the index changes. */
  static Status device_key(Device device, std::string& out) {
    int major = 0, minor = 0, bus = 0;
    if (attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                  device) != success() ||
        attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                  device) != success() ||
        attribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device) != success()) {
      return CUDA_ERROR_UNKNOWN;
    }
    out = "cuda:sm" + std::to_string(major * 10 + minor) + ":bus" +
          std::to_string(bus);
    return CUDA_SUCCESS;
  }

  static int shared_optin_limit(Device device) {
    int limit = 0;
    cuDeviceGetAttribute(
        &limit, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);
    return limit;
  }
  /* Below 48 KiB, dynamic shared memory needs no opt-in. */
  static unsigned shared_optin_threshold() { return 48u * 1024u; }
};

}  // namespace

extern "C" __attribute__((visibility("default"))) const FlagcvBackendV2*
flagcv_backend_v2() {
  return flagcv::driver::table<NvidiaApi>();
}
