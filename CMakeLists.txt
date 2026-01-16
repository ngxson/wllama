cmake_minimum_required(VERSION 3.14)
project("wllama")

option(GGML_WEBGPU "Enable GGML WebGPU backend" ON)
option(GGML_WEBGPU_JSPI "Enable GGML WebGPU JSPI support" ON)
option(LLAMA_WASM_MEM64 "Enable 64-bit memory for WebAssembly builds" OFF)
add_subdirectory(llama.cpp)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(WLLAMA_SRC cpp/wllama.cpp
    cpp/actions.hpp
    cpp/glue.hpp
    cpp/helpers/wlog.cpp
    cpp/helpers/wcommon.cpp
    cpp/helpers/wsampling.cpp
    llama.cpp/include/llama.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/helpers)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include)

add_executable(wllama ${WLLAMA_SRC})
target_link_libraries(wllama PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
