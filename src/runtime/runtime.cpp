/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/runtime/runtime.hpp"

#include <dlfcn.h>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "backends/backend.hpp"
#include "src/runtime/compiler_client.hpp"
#include "src/runtime/json.hpp"
#include "src/runtime/paths.hpp"

namespace flagcv {

namespace {

const char* depth_name(Depth depth) {
  switch (depth) {
    case Depth::k8U:
      return "8U";
    case Depth::k8S:
      return "8S";
    case Depth::k16U:
      return "16U";
    case Depth::k16S:
      return "16S";
    case Depth::k32S:
      return "32S";
    case Depth::k16F:
      return "16F";
    case Depth::k32F:
      return "32F";
    case Depth::k64F:
      return "64F";
  }
  throw Error("flagcv: unknown pixel depth");
}

/*
 * A loaded plugin. Plugins are opened once per process and never closed:
 * unloading a vendor runtime while its threads or atexit handlers are live is
 * a common source of shutdown crashes, and the mapping is small.
 */
const FlagcvBackendV2& load_plugin(const std::string& backend) {
  static std::mutex mutex;
  static std::map<std::string, const FlagcvBackendV2*> plugins;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = plugins.find(backend);
  if (it != plugins.end()) return *it->second;

  for (char c : backend) {
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) {
      throw Error("flagcv: invalid backend name '" + backend + "'");
    }
  }
  const std::string path = plugin_path(backend);
  void* library = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    throw Error("flagcv: cannot load backend plugin " + path + ": " +
                ::dlerror() + " (set FLAGCV_PLUGIN_DIR to override)");
  }
  auto entry = reinterpret_cast<FlagcvBackendEntry>(
      ::dlsym(library, FLAGCV_BACKEND_ENTRY));
  if (entry == nullptr)
    throw Error("flagcv: " + path + " does not export " FLAGCV_BACKEND_ENTRY);
  const FlagcvBackendV2* table = entry();
  if (table == nullptr || table->version != FLAGCV_BACKEND_VERSION) {
    throw Error("flagcv: " + path +
                " was built against a different plugin interface version");
  }
  plugins.emplace(backend, table);
  return *table;
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("flagcv: cannot read " + path);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

/* One grid dimension: a fixed extent, or ceil(product of params / block). */
struct GridDim {
  std::int64_t fixed = 1;
  std::vector<std::string> product;
  std::int64_t block = 1;
};

struct LoadedKernel {
  void* kernel = nullptr;
  unsigned int block_threads = 0;
  unsigned int shared_bytes = 0;
  GridDim grid[3];
};

/* Compiler identity per backend, asked once per process. */
const std::string& compiler_identity(const std::string& backend) {
  static std::mutex mutex;
  static std::map<std::string, std::string> identities;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = identities.find(backend);
  if (it == identities.end()) {
    json::Value reply = CompilerClient::instance().request(
        "\"op\":\"identify\",\"backend\":" + json::quote(backend));
    it = identities.emplace(backend, reply.at("identity").as_string()).first;
  }
  return it->second;
}

bool aligned16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
}

bool fits_i32(std::int64_t value) {
  return value >= std::numeric_limits<std::int32_t>::min() &&
         value <= std::numeric_limits<std::int32_t>::max();
}

/* Runtime params are specialised on width, and on divisibility by 16 when
 * the lowering asked for it (LoweredOp::Param::divisibility_hint). */
const char* runtime_type(std::int64_t value) {
  return fits_i32(value) ? "i32" : "i64";
}
bool divisible16(const LoweredOp::Param& param) {
  return param.divisibility_hint && param.value % 16 == 0;
}

}  // namespace

struct Handle::State {
  std::string backend;
  int device = 0;
  const FlagcvBackendV2* plugin = nullptr;
  void* context = nullptr;
  std::string device_id;
  std::string target;
  std::string compiler_identity;  // fetched on the first launch
  std::unordered_map<std::string, LoadedKernel> kernels;

  ~State() {
    // The cache goes before the plugin context it was loaded into.
    if (plugin == nullptr) return;
    for (auto& entry : kernels) plugin->unload(context, entry.second.kernel);
    kernels.clear();
    if (context != nullptr) plugin->destroy_context(context);
  }
};

struct HandleAccess {
  static Handle::State& state(Handle& handle) { return *handle.state_; }
};

Handle::Handle(const std::string& backend, int device) : state_(new State) {
  state_->backend = backend;
  state_->device = device;
  state_->plugin = &load_plugin(backend);
  const FlagcvBackendV2& plugin = *state_->plugin;
  if (plugin.create_context(device, &state_->context) != 0) {
    state_->plugin = nullptr;  // nothing to release
    throw Error(std::string("flagcv: ") + plugin.last_error());
  }
  char buffer[256];
  if (plugin.device_id(state_->context, buffer, sizeof(buffer)) != 0)
    throw Error(plugin.last_error());
  state_->device_id = buffer;
  if (plugin.target(state_->context, buffer, sizeof(buffer)) != 0)
    throw Error(plugin.last_error());
  state_->target = buffer;
}

Handle::~Handle() = default;
Handle::Handle(Handle&&) noexcept = default;
Handle& Handle::operator=(Handle&&) noexcept = default;

const std::string& Handle::backend() const { return state_->backend; }

ResolvedKernel resolve_kernel(const std::string& backend,
                              const std::string& kernel) {
  const std::string root = data_dir();
  const struct {
    std::string dir;
    bool platform;
  } tiers[] = {{root + "/backends/" + backend + "/kernels", true},
               {root + "/kernels", false}};

  for (const auto& tier : tiers) {
    const std::string registry_path = tier.dir + "/registry.json";
    std::ifstream probe(registry_path);
    if (!probe) {
      if (tier.platform) continue;  // a platform may ship no registry at all
      throw Error("flagcv: missing kernel registry " + registry_path);
    }
    json::Value registry = json::Value::parse(read_file(registry_path));
    for (const json::Value& entry : registry.at("kernels").as_array()) {
      if (entry.at("name").as_string() != kernel) continue;
      // Platform registration is exclusive: the first tier that names the
      // kernel wins, and its failures are never retried against the next.
      return ResolvedKernel{tier.dir + "/" + entry.at("source").as_string(),
                            entry.at("function").as_string(), tier.platform};
    }
  }
  throw Error("flagcv: no kernel registered for '" + kernel + "' on backend '" +
              backend + "'");
}

void launch(Handle& handle, const LoweredOp& op,
            const std::vector<void*>& pointers, Stream stream) {
  auto& state = HandleAccess::state(handle);
  const FlagcvBackendV2& plugin = *state.plugin;
  if (pointers.size() != op.images.size()) {
    throw Error("flagcv: launch got " + std::to_string(pointers.size()) +
                " pointers for " + std::to_string(op.images.size()) +
                " images");
  }

  // Cache key: everything that changes the compiled binary, nothing that
  // does not. Pixel values, extents and strides stay out; runtime params
  // contribute only their width. The compiler identity covers the kernel
  // sources, provider and toolchain versions.
  if (state.compiler_identity.empty())
    state.compiler_identity = compiler_identity(state.backend);
  std::ostringstream key;
  key << op.kernel;
  for (std::size_t i = 0; i < op.images.size(); ++i) {
    key << '|' << depth_name(op.images[i].depth) << 'c' << op.images[i].channels
        << (aligned16(pointers[i]) ? "a16" : "a1");
  }
  for (const auto& param : op.params) {
    key << '|' << param.name << '=';
    if (param.compile_time)
      key << param.value;
    else
      key << runtime_type(param.value) << (divisible16(param) ? ":16" : "");
  }
  key << '|' << state.device_id << '|' << state.compiler_identity;

  auto it = state.kernels.find(key.str());
  if (it == state.kernels.end()) {
    const ResolvedKernel resolved = resolve_kernel(state.backend, op.kernel);
    const std::string full_key = key.str();

    std::ostringstream body;
    body << "\"op\":\"compile\",\"backend\":" << json::quote(state.backend)
         << ",\"target\":" << json::quote(state.target)
         << ",\"kernel\":" << json::quote(op.kernel)
         << ",\"source\":" << json::quote(resolved.source)
         << ",\"function\":" << json::quote(resolved.function)
         << ",\"from_platform_registry\":"
         << (resolved.from_platform_registry ? "true" : "false")
         << ",\"images\":[";
    for (std::size_t i = 0; i < op.images.size(); ++i) {
      const ImageSpec& image = op.images[i];
      body << (i ? "," : "") << "{\"name\":" << json::quote(image.name)
           << ",\"depth\":\"" << depth_name(image.depth)
           << "\",\"channels\":" << image.channels
           << ",\"aligned16\":" << (aligned16(pointers[i]) ? "true" : "false")
           << "}";
    }
    body << "],\"params\":[";
    for (std::size_t i = 0; i < op.params.size(); ++i) {
      const auto& param = op.params[i];
      body << (i ? "," : "") << "{\"name\":" << json::quote(param.name);
      if (param.compile_time)
        body << ",\"type\":\"constexpr\",\"value\":" << param.value << "}";
      else
        body << ",\"type\":\"" << runtime_type(param.value)
             << "\",\"div16\":" << (divisible16(param) ? "true" : "false")
             << "}";
    }
    body << "]";

    json::Value reply = CompilerClient::instance().request(body.str());
    const std::string binary = read_file(reply.at("artifact").as_string());

    LoadedKernel loaded;
    const std::int64_t warp_size =
        std::stoll(state.target.substr(state.target.rfind(':') + 1));
    loaded.block_threads =
        static_cast<unsigned int>(reply.at("num_warps").as_int() * warp_size);
    loaded.shared_bytes =
        static_cast<unsigned int>(reply.at("shared").as_int());
    const auto& grid = reply.at("grid").as_array();
    if (grid.size() != 3)
      throw Error("flagcv: compiler returned a grid without three dimensions");
    for (int d = 0; d < 3; ++d) {
      if (grid[d].is_number()) {
        loaded.grid[d].fixed = grid[d].as_int();
        continue;
      }
      for (const auto& name : grid[d].at("product").as_array())
        loaded.grid[d].product.push_back(name.as_string());
      loaded.grid[d].block = grid[d].at("block").as_int();
    }
    const unsigned int flags =
        reply.has("load_flags")
            ? static_cast<unsigned int>(reply.at("load_flags").as_int())
            : 0u;
    if (plugin.load(state.context, binary.data(), binary.size(),
                    reply.at("function").as_string().c_str(),
                    loaded.shared_bytes, flags, &loaded.kernel) != 0) {
      throw Error(plugin.last_error());
    }
    it = state.kernels.emplace(full_key, loaded).first;
  }
  const LoadedKernel& kernel = it->second;

  unsigned int grid[3];
  for (int d = 0; d < 3; ++d) {
    std::int64_t extent = kernel.grid[d].fixed;
    if (!kernel.grid[d].product.empty()) {
      std::int64_t total = 1;
      for (const auto& name : kernel.grid[d].product) {
        bool found = false;
        for (const auto& param : op.params) {
          if (param.name == name) {
            total *= param.value;
            found = true;
            break;
          }
        }
        if (!found)
          throw Error("flagcv: grid refers to unknown param '" + name + "'");
      }
      extent = (total + kernel.grid[d].block - 1) / kernel.grid[d].block;
    }
    if (extent < 0 || extent > std::numeric_limits<unsigned int>::max()) {
      throw Error("flagcv: launch grid dimension out of range");
    }
    grid[d] = static_cast<unsigned int>(extent);
  }

  // Native launch convention: one pointer per kernel argument, to its value.
  // `kinds` records each argument's width in the same order: 0 for an 8-byte
  // value (device pointer or i64), 1 for a 4-byte one (i32). Only Ascend's
  // packed launch needs it; the others launch from the pointer array alone.
  std::vector<std::uint64_t> device_ptrs(pointers.size());
  std::vector<std::int64_t> wide;
  std::vector<std::int32_t> narrow;
  wide.reserve(op.params.size());
  narrow.reserve(op.params.size());
  std::vector<void*> args;
  std::vector<unsigned int> kinds;
  for (std::size_t i = 0; i < pointers.size(); ++i) {
    device_ptrs[i] = reinterpret_cast<std::uintptr_t>(pointers[i]);
    args.push_back(&device_ptrs[i]);
    kinds.push_back(0);
  }
  for (const auto& param : op.params) {
    if (param.compile_time) continue;  // baked into the binary, not an argument
    if (fits_i32(param.value)) {
      narrow.push_back(static_cast<std::int32_t>(param.value));
      args.push_back(&narrow.back());
      kinds.push_back(1);
    } else {
      wide.push_back(param.value);
      args.push_back(&wide.back());
      kinds.push_back(0);
    }
  }
  if (plugin.appends_scratch_args) {
    // Triton's ABI ends with two scratch pointers. The compiler rejects any
    // kernel that needs real scratch memory, so null is correct here.
    std::uint64_t global_scratch = 0, profile_scratch = 0;
    args.push_back(&global_scratch);
    args.push_back(&profile_scratch);
    kinds.push_back(0);
    kinds.push_back(0);
  }

  if (plugin.launch(state.context, kernel.kernel, grid, kernel.block_threads,
                    kernel.shared_bytes, stream, args.data(), kinds.data(),
                    static_cast<unsigned int>(args.size())) != 0) {
    throw Error(plugin.last_error());
  }
}

}  // namespace flagcv
