/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/backend.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void check(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

template <typename F>
int guard(flagcv::BackendError* error, F&& fn) noexcept {
  if (error) error->message[0] = '\0';
  try {
    fn();
    return 0;
  } catch (const std::exception& e) {
    if (error) std::snprintf(error->message, sizeof(error->message), "%s", e.what());
  } catch (...) {
    if (error) std::snprintf(error->message, sizeof(error->message), "unknown HIP error");
  }
  return 1;
}

// A backend call must not change the caller's current HIP device.
class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    check(hipGetDevice(&previous_), "hipGetDevice");
    check(hipSetDevice(device), "hipSetDevice");
  }
  ~DeviceScope() { (void)hipSetDevice(previous_); }
  DeviceScope(const DeviceScope&) = delete;
  DeviceScope& operator=(const DeviceScope&) = delete;

 private:
  int previous_ = 0;
};

struct Context {
  int device;
  std::string identity;
};

struct Kernel {
  Context* owner;
  hipModule_t module;
  hipFunction_t function;
};

Context& context_of(void* context) {
  if (!context) throw std::invalid_argument("AMD context is null");
  return *static_cast<Context*>(context);
}

Kernel& kernel_of(void* context, void* kernel) {
  if (!kernel || static_cast<Kernel*>(kernel)->owner != &context_of(context)) {
    throw std::invalid_argument("AMD kernel does not belong to this context");
  }
  return *static_cast<Kernel*>(kernel);
}

int create_context(int device, void** output, flagcv::BackendError* error) noexcept {
  if (output) *output = nullptr;
  return guard(error, [&] {
    if (!output) throw std::invalid_argument("context output is null");
    int count = 0;
    check(hipGetDeviceCount(&count), "hipGetDeviceCount");
    if (device < 0 || device >= count) {
      throw std::invalid_argument("AMD device index is out of range");
    }
    DeviceScope scope(device);
    auto context = std::make_unique<Context>();
    context->device = device;
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
    int runtime = 0;
    check(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
    char pci[64] = {};
    check(hipDeviceGetPCIBusId(pci, sizeof(pci), device), "hipDeviceGetPCIBusId");
    context->identity = "amd:" + std::string(pci) + ":" +
                        properties.gcnArchName + ":wave" +
                        std::to_string(properties.warpSize) + ":hip" +
                        std::to_string(runtime);
    // HIP owns device initialization. Do not create a separate context or
    // reset the device: allocations and streams belong to the caller.
    *output = context.release();
  });
}

int destroy_context(void* value, flagcv::BackendError* error) noexcept {
  return guard(error, [&] {
    delete &context_of(value);
  });
}

int device_id(void* value, char* output, std::size_t capacity,
              flagcv::BackendError* error) noexcept {
  return guard(error, [&] {
    const auto& identity = context_of(value).identity;
    if (!output || capacity <= identity.size()) {
      throw std::invalid_argument("device identity buffer is too small");
    }
    std::memcpy(output, identity.c_str(), identity.size() + 1);
  });
}

int load(void* value, const char* artifact, const char* entry, void** output,
         flagcv::BackendError* error) noexcept {
  if (output) *output = nullptr;
  return guard(error, [&] {
    auto& context = context_of(value);
    if (!output || !artifact || !entry || !*entry) {
      throw std::invalid_argument("invalid AMD module load arguments");
    }
    DeviceScope scope(context.device);
    auto kernel = std::make_unique<Kernel>();
    kernel->owner = &context;
    check(hipModuleLoad(&kernel->module, artifact), "hipModuleLoad");
    const auto status = hipModuleGetFunction(&kernel->function, kernel->module, entry);
    if (status != hipSuccess) {
      (void)hipModuleUnload(kernel->module);
      check(status, "hipModuleGetFunction");
    }
    *output = kernel.release();
  });
}

int unload(void* context, void* value, flagcv::BackendError* error) noexcept {
  return guard(error, [&] {
    auto& kernel = kernel_of(context, value);
    DeviceScope scope(kernel.owner->device);
    check(hipModuleUnload(kernel.module), "hipModuleUnload");
    delete &kernel;
  });
}

int launch(void* context, void* value, const flagcv::LaunchConfig* config,
           void** arguments, void* stream, flagcv::BackendError* error) noexcept {
  return guard(error, [&] {
    auto& kernel = kernel_of(context, value);
    if (!config) throw std::invalid_argument("launch configuration is null");
    for (int i = 0; i < 3; ++i) {
      if (!config->grid[i] || !config->block[i]) {
        throw std::invalid_argument("launch dimensions must be nonzero");
      }
    }
    DeviceScope scope(kernel.owner->device);
    check(hipModuleLaunchKernel(
              kernel.function, config->grid[0], config->grid[1], config->grid[2],
              config->block[0], config->block[1], config->block[2],
              config->shared_memory_bytes, reinterpret_cast<hipStream_t>(stream),
              arguments, nullptr),
          "hipModuleLaunchKernel");
  });
}

const flagcv::BackendApi api = {
    FLAGCV_BACKEND_VERSION, "amd", create_context, destroy_context,
    device_id, load, unload, launch};

}  // namespace

extern "C" const flagcv::BackendApi* flagcv_backend_api() { return &api; }
