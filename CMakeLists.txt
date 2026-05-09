cmake_minimum_required(VERSION 3.14)
project("wllama")

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

add_compile_options(
    -O3 -msimd128 -DNDEBUG
    -frtti -fwasm-exceptions
    -pthread
    -sMEMORY64=1
)
add_link_options(
    -sMEMORY64=1
    -fwasm-exceptions
    # pthread wraps
    -Wl,--wrap,pthread_mutex_init
    -Wl,--wrap,__pthread_mutex_lock
    -Wl,--wrap,__pthread_mutex_unlock
    -Wl,--wrap,__pthread_mutex_trylock
    -Wl,--wrap,__pthread_mutex_timedlock
    -Wl,--wrap,pthread_mutex_destroy
    -Wl,--wrap,pthread_mutex_consistent
    -Wl,--wrap,pthread_barrier_init
    -Wl,--wrap,pthread_barrier_destroy
    -Wl,--wrap,pthread_barrier_wait
    -Wl,--wrap,__pthread_create
    -Wl,--wrap,__pthread_join
    -Wl,--wrap,__pthread_detach
    -Wl,--wrap,pthread_cancel
    -Wl,--wrap,__pthread_testcancel
    -Wl,--wrap,__pthread_exit
    -Wl,--wrap,pthread_equal
    -Wl,--wrap,pthread_kill
    -Wl,--wrap,pthread_setcancelstate
    -Wl,--wrap,pthread_setcanceltype
    -Wl,--wrap,__pthread_key_create
    -Wl,--wrap,__pthread_key_delete
    -Wl,--wrap,pthread_getspecific
    -Wl,--wrap,pthread_setspecific
    -Wl,--wrap,__pthread_once
    -Wl,--wrap,pthread_cond_init
    -Wl,--wrap,pthread_cond_destroy
    -Wl,--wrap,pthread_cond_wait
    -Wl,--wrap,__pthread_cond_timedwait
    -Wl,--wrap,pthread_cond_signal
    -Wl,--wrap,pthread_cond_broadcast
    -Wl,--wrap,pthread_atfork
    -Wl,--wrap,pthread_rwlock_init
    -Wl,--wrap,pthread_rwlock_destroy
    -Wl,--wrap,pthread_rwlock_rdlock
    -Wl,--wrap,pthread_rwlock_tryrdlock
    -Wl,--wrap,pthread_rwlock_timedrdlock
    -Wl,--wrap,pthread_rwlock_wrlock
    -Wl,--wrap,pthread_rwlock_trywrlock
    -Wl,--wrap,pthread_rwlock_timedwrlock
    -Wl,--wrap,pthread_rwlock_unlock
    -Wl,--wrap,sem_post
    -Wl,--wrap,sem_wait
    -Wl,--wrap,sem_trywait
    --no-entry
    -sEXPORT_ALL=1
    -sEXPORT_ES6=0
    -sMODULARIZE=0
    -sINITIAL_MEMORY=128MB
    -sMAXIMUM_MEMORY=4096MB
    -sALLOW_MEMORY_GROWTH=1
    -sFORCE_FILESYSTEM=1
    -sEXPORTED_FUNCTIONS=_main,_wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug
    -sEXPORTED_RUNTIME_METHODS=ccall,cwrap
    -sNO_EXIT_RUNTIME=1
    -sIMPORTED_MEMORY=1
    -sPTHREAD_POOL_SIZE=Module[\"pthreadPoolSize\"]
    -sUSE_PTHREADS=1
    -pthread
)

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
    cpp/wllama-pthread.cpp
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
