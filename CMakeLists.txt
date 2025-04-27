cmake_minimum_required(VERSION 3.14)
project("wllama")
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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/examples/llava)

#
# multimodal
#
add_library(mtmd OBJECT
            llama.cpp/examples/llava/mtmd.cpp
            llama.cpp/examples/llava/mtmd.h
            llama.cpp/examples/llava/clip.cpp
            llama.cpp/examples/llava/clip.h
            llama.cpp/examples/llava/clip-impl.h
            )

target_link_libraries(mtmd PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(mtmd PUBLIC  llama.cpp/examples/llava)
target_include_directories(mtmd PRIVATE llama.cpp)
target_include_directories(mtmd PRIVATE llama.cpp/common) # for stb_image.h

target_compile_features(mtmd PRIVATE cxx_std_17)
target_compile_options(mtmd PRIVATE -Wno-cast-qual) # stb_image.h
target_compile_options(mtmd PRIVATE -Wno-c++11-narrowing) # clip.cpp

add_library(mtmd_static STATIC $<TARGET_OBJECTS:mtmd>)

#
# wllama target definition
#
add_executable(wllama ${WLLAMA_SRC})
target_link_libraries(wllama PRIVATE ggml llama mtmd_static ${CMAKE_THREAD_LIBS_INIT})
