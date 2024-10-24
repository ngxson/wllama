cmake_minimum_required(VERSION 3.14)
project("wllama")
add_subdirectory(llama.cpp)
add_subdirectory(llama.cpp/common)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(COMMON_SRC actions.hpp
    json.hpp
    llama.cpp/include/llama.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common)

add_executable(wllama wllama.cpp ${COMMON_SRC})
target_link_libraries(wllama PRIVATE ggml llama common ${CMAKE_THREAD_LIBS_INIT})
