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

inline std::string deviceStr(const c10::Device &device) {
  auto s = device.str();
  if (s == "cuda") {
    auto curr_dev = at::cuda::current_device();
    return "cuda:" + std::to_string(curr_dev);
  } else {
    return s;
  }
}

inline c10::optional<std::string> jsonIValue(
  const c10::IValue& val,
  const size_t maxArrayLen = 4096) {
  if (val.isTensor()) {
    const auto t = val.toTensor();
    if (t.has_storage()) {
      auto shape = vectorToString(t.sizes().vec());
      auto dtype = t.dtype().toScalarType();
      auto device = deviceStr(t.device());
      return concat("{",
        "\"type\":", "\"Tensor\",",
        "\"shape\":", shape, ",",
        "\"dtype\":", static_cast<int32_t>(dtype), ",",
        "\"device\":", "\"", device, "\"",
      "}");
    } else {
      return c10::nullopt;
    }
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
    if (std::isinf(d_val)) {
      if (d_val > 0) {
        d_val = std::numeric_limits<double>::max();
      } else {
        d_val = -std::numeric_limits<double>::max();
      }
    }
    if (std::isnan(d_val)) {
      d_val = 0;
    }
    return concat("{",
      "\"type\":", "\"Double\",",
      "\"value\":", d_val,
    "}");
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
      "\"value\":", "\"", deviceStr(val.toDevice()), "\"",
    "}");
  }
  return c10::nullopt;
}

inline std::string jsonStream(cudaStream_t stream) {
  struct _cudaStream {
    int device;
    int id;
  };
  if (stream) {
    auto stream_ = (_cudaStream *)stream;
    return concat("[",
      stream_->device, ",",
      stream_->id,
    "]");
  } else {
    return "null";
  }
}

void sendOneCall(
    int simulator_sock_fd,
    size_t id, size_t parent,
    long cur_sim_time,
    const char* name,
    const std::vector<std::string>& args) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  static char HOSTNAME_BUF[256];
  gethostname(HOSTNAME_BUF, sizeof(HOSTNAME_BUF));
  // \x02 tag for torch call message
  auto info = concat("{",
    "\"pid\":", getpid(), ",",
    "\"hostname\":", "\"", HOSTNAME_BUF, "\",",
    "\"stream\":", jsonStream(stream), ",",
    "\"id\":", id, ",",
    "\"parent\":", parent, ",",
    "\"cur\":", cur_sim_time, ",",
    "\"name\":", "\"", name, "\",",
    "\"args\":", vectorToString(args),
  "}\x02");

  int ret = send(simulator_sock_fd, info.c_str(), info.size(), 0);
  if (ret < 0) {
    LOG(WARNING) << "Failed to send torch call to simulator: " << strerror(errno);
  }
}

static inline long current_time_us() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto cur_time = std::chrono::duration_cast<std::chrono::microseconds>(now);
  return cur_time.count();
}

struct TORCH_API FunctionTracer {
  int simulator_sock_fd{-1};
  std::mutex g_mutex{};
  CallbackHandle cb_handle{INVALID_CALLBACK_HANDLE};
  size_t next_id{1};
  std::stack<size_t> call_stack{};
  void* cudalib_handle{nullptr}; // The preloaded library
  long (*get_time_offset)(){nullptr};
  void (*subtract_time)(long){nullptr};

  FunctionTracer() = default;
};

using TracerManager = GlobalStateManager<FunctionTracer>;

std::unique_ptr<ObserverContext> tracerOnFunctionEnter(const RecordFunction& fn) {
  auto tracer = TracerManager::get();
  if (tracer != nullptr) {
    try {
      const std::lock_guard<std::mutex> lock(tracer->g_mutex);

      auto start_time = current_time_us();
      auto cur_sim_time = start_time + tracer->get_time_offset();

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

      sendOneCall(
          tracer->simulator_sock_fd,
          id, parent, cur_sim_time,
          fn.name(), args);

      auto end_time = current_time_us();
      tracer->subtract_time(end_time - start_time);
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
      tracer->get_time_offset = (long (*)())get_time_offset;
    }

    auto subtract_time = dlsym(cudalib_handle, "subtract_time");
    if (subtract_time == nullptr) {
      LOG(WARNING) << "Failed to find subtract_time in libcuda.so.1: " << dlerror();
    } else {
      tracer->subtract_time = (void (*)(long))subtract_time;
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
