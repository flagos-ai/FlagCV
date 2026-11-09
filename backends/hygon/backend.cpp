/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <hip/hip_runtime.h>

#include "backends/common/driver_backend.hpp"

/*
 * Hygon (DCU) plugin.
 *
 * The HIP module API has the CUDA driver shape, so the shared flow applies;
 * the compile target is Triton's "hip" backend with the GPU architecture
 * (gfx942 and so on), and the artifact is a .hsaco.
 *
 * Includes no core internal header.
 */

namespace {

struct HygonApi {
  using Status = hipError_t;
  using Device = hipDevice_t;
  using Context = hipCtx_t;
  using Module = hipModule_t;
  using Function = hipFunction_t;

  static const char* name() { return "hygon"; }
  static Status success() { return hipSuccess; }

  static Status init() { return hipInit(0); }
  static Status device_get(Device* device, int index) {
    return hipDeviceGet(device, index);
  }
  static Status primary_ctx_retain(Context* ctx, Device device) {
    return hipDevicePrimaryCtxRetain(ctx, device);
  }
  static Status primary_ctx_release(Device device) {
    return hipDevicePrimaryCtxRelease(device);
  }
  static Status ctx_push(Context ctx) { return hipCtxPushCurrent(ctx); }
  static Status ctx_pop(Context* ctx) { return hipCtxPopCurrent(ctx); }
  static Status module_load_data(Module* module, const void* data) {
    return hipModuleLoadData(module, data);
  }
  static Status module_get_function(Function* fn, Module module,
                                    const char* name) {
    return hipModuleGetFunction(fn, module, name);
  }
  static Status module_unload(Module module) { return hipModuleUnload(module); }
  static Status launch_kernel(Function fn, unsigned gx, unsigned gy,
                              unsigned gz, unsigned block, unsigned shared,
                              void* stream, void** args) {
    // hipModuleLaunchKernel takes the stream by type and the extra
    // shared-memory/stream arguments Triton does not use.
    return hipModuleLaunchKernel(fn, gx, gy, gz, block, 1, 1, shared,
                                 static_cast<hipStream_t>(stream), args,
                                 nullptr);
  }
  static Status set_shared(Function fn, int bytes) {
    return hipFuncSetAttribute(fn, hipFuncAttributeMaxDynamicSharedMemorySize,
                               bytes);
  }

  static Status attribute(int* value, hipDeviceAttribute_t what,
                          Device device) {
    return hipDeviceGetAttribute(value, what, device);
  }

  static Status target(Device device, std::string& out) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device) != hipSuccess)
      return hipErrorUnknown;
    // gcnArchName carries target features ("gfx936:sramecc+:xnack-"); Triton
    // takes the bare architecture, as its own HIP driver does.
    std::string arch = props.gcnArchName;
    out = "hip:" + arch.substr(0, arch.find(':')) + ":" +
          std::to_string(props.warpSize);
    return hipSuccess;
  }

  static Status device_key(Device device, std::string& out) {
    hipDeviceProp_t props{};
    int bus = 0;
    if (hipGetDeviceProperties(&props, device) != hipSuccess)
      return hipErrorUnknown;
    hipDeviceGetAttribute(&bus, hipDeviceAttributePciBusId, device);
    out =
        std::string("hip:") + props.gcnArchName + ":bus" + std::to_string(bus);
    return hipSuccess;
  }

  static int shared_optin_limit(Device device) {
    // DTK's HIP has no MaxSharedMemoryPerBlockOptin attribute and its legacy
    // hipDeviceProp_t lacks the field; only the v2 properties carry it.
    hipDeviceProp_t_v2 props{};
    if (hipGetDeviceProperties_v2(&props, device) != hipSuccess) return 0;
    return static_cast<int>(props.sharedMemPerBlockOptin);
  }
  /* HIP exposes the opt-in limit directly; keep the CUDA convention. */
  static unsigned shared_optin_threshold() { return 48u * 1024u; }
};

}  // namespace

extern "C" __attribute__((visibility("default"))) const FlagcvBackendV2*
flagcv_backend_v2() {
  return flagcv::driver::table<HygonApi>();
}
