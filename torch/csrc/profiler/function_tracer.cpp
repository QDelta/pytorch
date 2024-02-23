#include "c10/cuda/CUDAFunctions.h"
#ifdef _WIN32
#error "Not supported on Windows"
#else
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <dlfcn.h>
#endif

#include <chrono>
#include <cmath>
#include <mutex>
#include <stack>
#include <vector>

#include <ATen/core/TensorBody.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/stack.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/profiler/function_tracer.h>
#include <torch/csrc/profiler/util.h>

using namespace at;

namespace torch {
namespace profiler {
namespace impl {

template<typename T>
inline std::string vectorToString(const std::vector<T>& v) {
  std::ostringstream os;
  os << "[";
  if (!v.empty()) {
    os << v[0];
    for (const auto i : c10::irange(1, v.size())) {
      os << "," << v[i];
    }
  }
  os << "]";
  return os.str();
}

inline void output_all(std::ostringstream& os) {}

template<typename Arg, typename... Args>
inline void output_all(std::ostringstream& os, Arg arg, Args... args) {
  os << arg;
  output_all(os, args...);
}

template<typename... Args>
inline std::string concat(Args... args) {
  std::ostringstream os;
  output_all(os, args...);
  return os.str();
}

constexpr size_t maxNumElements = 4096;

inline c10::optional<std::string> jsonIValue(
  const c10::IValue& val,
  const size_t maxArrayLen = maxNumElements) {
  if (val.isTensor()) {
    const auto t = val.toTensor();
    auto shape = vectorToString(t.sizes().vec());
    auto dtype = std::string(t.dtype().name());
    auto device = t.device().str();
    return concat("{",
      "\"type\":", "\"Tensor\",",
      "\"shape\":", shape, ",",
      "\"dtype\":", "\"", dtype, "\",",
      "\"device\":", "\"", device, "\"",
    "}");
  } else if (val.isTuple()) {
    std::vector<std::string> element_jsons;
    const auto& elements = val.toTupleRef().elements();
    for (const auto j: c10::irange(elements.size())) {
      const auto e_json = jsonIValue(elements[j], maxArrayLen);
      if (e_json.has_value()) {
        element_jsons.emplace_back(e_json.value());
      }
    }
    return concat("{",
      "\"type\":", "\"Tuple\",",
      "\"elements\":", vectorToString(element_jsons),
    "}");
  } else if (val.isList()) {
    std::vector<std::string> element_jsons;
    const auto& elements = val.toList();
    for (const auto j: c10::irange(elements.size())) {
      const auto e_json = jsonIValue(elements.get(j), maxArrayLen);
      if (e_json.has_value()) {
        element_jsons.emplace_back(e_json.value());
      }
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << elements.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;
        break;
      }
    }
    return concat("{",
      "\"type\":", "\"List\",",
      "\"elements\":", vectorToString(element_jsons),
    "}");
  } else if (val.isDouble()) {
    double d_val = val.toDouble();
    if (std::isinf(d_val) || std::isnan(d_val)) {
      return concat("{",
        "\"type\":", "\"Double\",",
        "\"value\":", "\"", std::to_string(d_val), "\"",
      "}");
    } else { 
      return concat("{",
        "\"type\":", "\"Double\",",
        "\"value\":", d_val,
      "}");
    }
  } else if (val.isInt()) {
    return concat("{",
      "\"type\":", "\"Int\",",
      "\"value\":", val.toInt(),
    "}");
  } else if (val.isBool()) {
    return concat("{",
      "\"type\":", "\"Bool\",",
      "\"value\":", val.toBool() ? "true" : "false",
    "}");
  } else if (val.isString()) {
    const std::string& str_val = val.toStringRef();
    if (str_val.size() > maxArrayLen) {
      LOG(WARNING) << "string size=" << str_val.size()
                   << " exceeded maxArrayLen=" << maxArrayLen;
      return concat("{",
        "\"type\":", "\"String\",",
        "\"value\":", "\"", str_val.substr(0, maxArrayLen), "\"",
      "}");
    }
    return concat("{",
      "\"type\":", "\"String\",",
      "\"value\":", "\"", str_val, "\"",
    "}");
  } else if (val.isDevice()) {
    return concat("{",
      "\"type\":", "\"Device\",",
      "\"value\":", "\"", val.toDevice().str(), "\"",
    "}");
  }
  return c10::nullopt;
}

void sendOneCall(
    int simulator_sock_fd,
    size_t id, size_t parent,
    long long cur_sim_nanos,
    const char* name,
    const std::vector<std::string>& args) {
  int device = at::cuda::current_device();
  int stream_id = 0;
  struct deepsim_cudaStream {
    int device;
    int id;
  };
  auto stream = (deepsim_cudaStream *)(at::cuda::getCurrentCUDAStream().stream());
  if (stream) {
    device = stream->device;
    stream_id = stream->id;
  }
  // \x02 tag for torch call message
  auto info = concat("{",
    "\"dev\":", device, ",",
    "\"stream\":", stream_id, ",",
    "\"id\":", id, ",",
    "\"parent\":", parent, ",",
    "\"cur\":", cur_sim_nanos, ",",
    "\"name\":", "\"", name, "\",",
    "\"args\":", vectorToString(args),
  "}\x02");

  int ret = send(simulator_sock_fd, info.c_str(), info.size(), 0);
  if (ret < 0) {
    LOG(WARNING) << "Failed to send torch call to simulator: " << strerror(errno);
  }
}

struct TORCH_API FunctionTracer {
  int simulator_sock_fd{-1};
  std::mutex g_mutex{};
  CallbackHandle cb_handle{INVALID_CALLBACK_HANDLE};
  int32_t pid{-1};
  size_t next_id{1};
  std::stack<size_t> call_stack{};
  void* cudalib_handle{nullptr};
  long long (*get_time_offset)(){nullptr};

  FunctionTracer() = default;
};

using TracerManager = GlobalStateManager<FunctionTracer>;

std::unique_ptr<ObserverContext> tracerOnFunctionEnter(const RecordFunction& fn) {
  auto tracer = TracerManager::get();
  if (tracer != nullptr) {
    try {
      const std::lock_guard<std::mutex> lock(tracer->g_mutex);

      const auto num_inputs = fn.num_inputs();
      const auto inputs = fn.inputs();
      const auto size_inputs = inputs.size();
      std::vector<std::string> args;

      if (num_inputs > size_inputs) {
        LOG(WARNING) << "RecordFunction " << fn.name()
                     << " expected num_inputs=" << num_inputs
                     << " > inputs.size()=" << size_inputs;
      } else {
        for (const auto i : c10::irange(size_inputs - num_inputs, size_inputs)) {
          const auto arg_json = jsonIValue(inputs[i]);
          if (arg_json.has_value()) {
            args.emplace_back(arg_json.value());
          }
        }
      }

      size_t id = tracer->next_id++;
      size_t parent = tracer->call_stack.top();
      tracer->call_stack.push(id);

      auto now = std::chrono::system_clock::now().time_since_epoch();
      auto cur_nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now);
      auto cur_sim_nanos = cur_nanos.count() + tracer->get_time_offset();

      sendOneCall(
          tracer->simulator_sock_fd,
          id, parent, cur_sim_nanos,
          fn.name(), args);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Exception in function tracer (enter): " << e.what();
    }
  }
  return nullptr;
}

void tracerOnFunctionExit(const RecordFunction& fn, ObserverContext* ctx_ptr) {
  auto tracer = TracerManager::get();
  if (tracer != nullptr) {
    try {
      const std::lock_guard<std::mutex> lock(tracer->g_mutex);
      tracer->call_stack.pop();
    } catch (const std::exception& e) {
      LOG(WARNING) << "Exception in function tracer (exit): " << e.what();
    }
  }
}

void enableFunctionTracer(const std::string& simulator_sock_path) {
  auto tracer = TracerManager::get();
  if (tracer == nullptr) {
    TracerManager::push(std::make_shared<FunctionTracer>());
    tracer = TracerManager::get();
  } else if (tracer->cb_handle != INVALID_CALLBACK_HANDLE) {
    LOG(WARNING) << "Function tracer was already enabled.";
    return;
  }
  tracer->pid = getpid();
  tracer->call_stack.push(0);

  int sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  auto simulator_addr = (sockaddr_un*)malloc(sizeof(sockaddr_un));
  simulator_addr->sun_family = AF_UNIX;
  strncpy(simulator_addr->sun_path, simulator_sock_path.c_str(), sizeof(simulator_addr->sun_path) - 1);
  int ret = connect(sock_fd, (sockaddr*)simulator_addr, sizeof(sockaddr_un));
  if (ret < 0) {
    LOG(WARNING) << "Failed to connect to simulator: " << strerror(errno);
    return;
  }
  free(simulator_addr);
  tracer->simulator_sock_fd = sock_fd;

  tracer->cb_handle = addGlobalCallback(
      RecordFunctionCallback(&tracerOnFunctionEnter, &tracerOnFunctionExit)
          .needsInputs(true)
          .needsOutputs(true)
          .needsIds(true));
  
  auto cudalib_handle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (cudalib_handle == nullptr) {
    LOG(WARNING) << "Failed to open libcuda.so.1: " << dlerror();
  } else {
    tracer->cudalib_handle = cudalib_handle;
    auto get_time_offset = dlsym(cudalib_handle, "get_time_offset");
    if (get_time_offset == nullptr) {
      LOG(WARNING) << "Failed to find get_time_offset in libcuda.so.1: " << dlerror();
    } else {
      tracer->get_time_offset = (long long (*)())get_time_offset;
    }
  }
}

void disableFunctionTracer() {
  auto tracer = TracerManager::get();
  if (tracer != nullptr) {
    close(tracer->simulator_sock_fd);
    removeCallback(tracer->cb_handle);
    tracer->cb_handle = INVALID_CALLBACK_HANDLE;
    if (tracer->cudalib_handle != nullptr) {
      dlclose(tracer->cudalib_handle);
    }
  } else {
    LOG(WARNING) << "Function tracer was not enabled.";
  }
}

} // namespace impl
} // namespace profiler
} // namespace torch