cmake_minimum_required(VERSION 3.14)
project("wllama")

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(WLLAMA_COMPILE_OPTIONS
    -O3 -msimd128 -DNDEBUG
    -flto=full -frtti
    -pthread
)
set(WLLAMA_LINK_OPTIONS
    -flto=full
    --no-entry
    -sEXPORT_ES6=0
    -sMODULARIZE=0
    -sINITIAL_MEMORY=128MB
    -sMAXIMUM_MEMORY=4096MB
    -sALLOW_MEMORY_GROWTH=1
    -sFORCE_FILESYSTEM=1
    -sEXPORTED_FUNCTIONS=_main,_wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug
    -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,MEMFS,FS,mmapAlloc,ENV,wasmMemory
    -sNO_EXIT_RUNTIME=1
    -sIMPORTED_MEMORY=1
    -sPTHREAD_POOL_SIZE=Module[\"pthreadPoolSize\"]
    -sUSE_PTHREADS=1
    -pthread
    -Wl,--wrap,fopen
    -Wl,--wrap,fclose
    -Wl,--wrap,fread
    -Wl,--wrap,fseek
    -Wl,--wrap,ftell
)

if (WLLAMA_COMPAT)
    # no wasm exception (not compatible with asyncify - asyncify is needed for firefox and safari)
    # no mem64 (not compatible with safari)
    list(APPEND WLLAMA_COMPILE_OPTIONS
        -fexceptions
        -pthread
    )
    list(APPEND WLLAMA_LINK_OPTIONS
        -fexceptions
        -sASYNCIFY=1
        -sASYNCIFY_ADD=['wllama_start','wllama_action']
    )
else()
    list(APPEND WLLAMA_COMPILE_OPTIONS
        -sMEMORY64=1
        -fwasm-exceptions
    )
    list(APPEND WLLAMA_LINK_OPTIONS
        -sMEMORY64=1
        -fwasm-exceptions
        -sJSPI
        -sJSPI_EXPORTS=['wllama_start','wllama_action']
    )
endif()

add_compile_options(${WLLAMA_COMPILE_OPTIONS})
add_link_options(${WLLAMA_LINK_OPTIONS})

add_subdirectory(llama.cpp)

set(LLAMA_INSTALL_VERSION 0.0.${LLAMA_BUILD_NUMBER})
add_subdirectory(llama.cpp/tools/mtmd)

file(GLOB_RECURSE LLAMA_COMMON_SRC
    CONFIGURE_DEPENDS
    llama.cpp/common/*.cpp
    llama.cpp/common/*.h
    llama.cpp/common/*.hpp
    llama.cpp/common/jinja/*.cpp
    llama.cpp/common/jinja/*.h
)

list(REMOVE_ITEM LLAMA_COMMON_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/arg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/console.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/fit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/hf-cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/log.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/preset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/download.cpp
)

set(LLAMA_SERVER_SRC llama.cpp/tools/server/server-context.cpp
    llama.cpp/tools/server/server-task.cpp
    llama.cpp/tools/server/server-chat.cpp
    llama.cpp/tools/server/server-common.cpp
)

set(WLLAMA_SRC ${LLAMA_COMMON_SRC} ${LLAMA_SERVER_SRC}
    cpp/wllama.cpp
    cpp/glue.hpp
    llama.cpp/include/llama.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/server)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd)

add_executable(wllama ${WLLAMA_SRC})
target_link_libraries(wllama PRIVATE ggml llama mtmd ${CMAKE_THREAD_LIBS_INIT})
