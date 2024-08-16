list(APPEND CMAKE_PREFIX_PATH $ENV{GRPC_INSTALL_PREFIX})
list(APPEND CMAKE_PREFIX_PATH $ENV{PROTOBUF_INSTALL_PREFIX})
list(APPEND CMAKE_MODULE_PATH $ENV{GRPC_MODULE_PATH})

message("CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
message("CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")
# find_package(Protobuf CONFIG REQUIRED PATHS ${CMAKE_PREFIX_PATH} NO_DEFAULT_PATH)
# find_package(gRPC CONFIG REQUIRED PATHS "/home/vscode/grpc/dist" NO_DEFAULT_PATH)
# find_package(Protobuf CONFIG REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Protobuf CONFIG REQUIRED)
if(NOT ${Protobuf_FOUND})
  message(FATAL_ERROR "Could not locate protobuf.")
endif()
find_package(gRPC CONFIG REQUIRED)
if(NOT ${gRPC_FOUND})
  message(FATAL_ERROR "Could not locate gRPC.")
endif()

get_target_property(Protobuf_INCLUDE_DIRS protobuf::libprotobuf
  INTERFACE_INCLUDE_DIRECTORIES)
# set(Protobuf_INCLUDE_DIRS "/home/vscode/dist/include")
get_target_property(gRPC_INCLUDE_DIRS gRPC::grpc
  INTERFACE_INCLUDE_DIRECTORIES)
message("Protobuf_INCLUDE_DIRS=${Protobuf_INCLUDE_DIRS}")
message("gRPC_INCLUDE_DIRS=${gRPC_INCLUDE_DIRS}")

# include_directories(${CMAKE_PREFIX_PATH}/include)
# target_include_directories(torch_cuda PUBLIC ${Protobuf_INCLUDE_DIRS})
# target_include_directories(torch_cuda PUBLIC ${gRPC_INCLUDE_DIRS})

# Generate source files from proto files
set(PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/gpusynth.proto)
set(GPUSYNTH_PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
message("PROTO_FILES=${PROTO_FILES}")
add_library(function_tracer_proto ${PROTO_FILES})
target_include_directories(function_tracer_proto INTERFACE ${Protobuf_INCLUDE_DIRS})
target_include_directories(function_tracer_proto INTERFACE ${gRPC_INCLUDE_DIRS})
# include_directories(${Protobuf_INCLUDE_DIRS})
# include_directories(${gRPC_INCLUDE_DIRS})
protobuf_generate(
  LANGUAGE cpp
  TARGET function_tracer_proto
  PROTOS ${PROTO_FILES}
  PROTOC_OUT_DIR ${GPUSYNTH_PROTO_BINARY_DIR}
  PROTOC_OPTIONS "--experimental_allow_proto3_optional"
)
protobuf_generate(
  LANGUAGE grpc
  TARGET function_tracer_proto
  GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
  PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
  PROTOS ${PROTO_FILES}
  PROTOC_OUT_DIR ${GPUSYNTH_PROTO_BINARY_DIR}
  PROTOC_OPTIONS "--experimental_allow_proto3_optional"
)
install(TARGETS function_tracer_proto
        EXPORT Caffe2Targets
        DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# Link protobuf and gRPC libraries
set(Protobuf_LIBRARIES "protobuf::libprotobuf")
set(gRPC_LIBRARIES "gRPC::grpc++")
message("==Protobuf_LIBRARIES=${Protobuf_LIBRARIES}")
message("==gRPC_LIBRARIES=${gRPC_LIBRARIES}")
target_link_libraries(function_tracer_proto PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(function_tracer_proto PUBLIC ${gRPC_LIBRARIES})

set(Protobuf_INCLUDE_DIRS ${gRPC_INCLUDE_DIRS} PARENT_SCOPE)
set(gRPC_INCLUDE_DIRS ${gRPC_INCLUDE_DIRS} PARENT_SCOPE)
set(GPUSYNTH_PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
set(Protobuf_LIBRARIES ${Protobuf_LIBRARIES} PARENT_SCOPE)
set(gRPC_LIBRARIES ${gRPC_LIBRARIES} PARENT_SCOPE)