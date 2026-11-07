cmake_minimum_required(VERSION 3.14)
project(wllama-native-test CXX C)

set(CMAKE_CXX_STANDARD 17)

# repo root, relative to this directory (dev/native-test)
get_filename_component(WLLAMA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

add_subdirectory(${WLLAMA_ROOT}/llama.cpp llama.cpp-build)
add_subdirectory(${WLLAMA_ROOT}/llama.cpp/tools/mtmd mtmd-build)

# same source list as the root CMakeLists.txt (the wasm build), minus wllama.cpp
file(GLOB_RECURSE LLAMA_COMMON_SRC
    CONFIGURE_DEPENDS
    ${WLLAMA_ROOT}/llama.cpp/common/*.cpp
    ${WLLAMA_ROOT}/llama.cpp/common/jinja/*.cpp
)
list(REMOVE_ITEM LLAMA_COMMON_SRC
    ${WLLAMA_ROOT}/llama.cpp/common/arg.cpp
    ${WLLAMA_ROOT}/llama.cpp/common/console.cpp
    ${WLLAMA_ROOT}/llama.cpp/common/debug.cpp
    ${WLLAMA_ROOT}/llama.cpp/common/fit.cpp
    ${WLLAMA_ROOT}/llama.cpp/common/hf-cache.cpp
    ${WLLAMA_ROOT}/llama.cpp/common/log.cpp
    ${WLLAMA_ROOT}/llama.cpp/common/preset.cpp
    ${WLLAMA_ROOT}/llama.cpp/common/download.cpp
)

set(LLAMA_SERVER_SRC
    ${WLLAMA_ROOT}/llama.cpp/tools/server/server-context.cpp
    ${WLLAMA_ROOT}/llama.cpp/tools/server/server-task.cpp
    ${WLLAMA_ROOT}/llama.cpp/tools/server/server-chat.cpp
    ${WLLAMA_ROOT}/llama.cpp/tools/server/server-common.cpp
    ${WLLAMA_ROOT}/llama.cpp/tools/server/server-schema.cpp
    ${WLLAMA_ROOT}/llama.cpp/tools/server/server-stream.cpp
)

include_directories(${WLLAMA_ROOT}/cpp)
include_directories(${WLLAMA_ROOT}/llama.cpp/include)
include_directories(${WLLAMA_ROOT}/llama.cpp/common)
include_directories(${WLLAMA_ROOT}/llama.cpp/vendor)
include_directories(${WLLAMA_ROOT}/llama.cpp/tools/server)
include_directories(${WLLAMA_ROOT}/llama.cpp/tools/mtmd)

add_executable(native-test main.cpp ${LLAMA_COMMON_SRC} ${LLAMA_SERVER_SRC})
target_link_libraries(native-test PRIVATE ggml llama mtmd)
