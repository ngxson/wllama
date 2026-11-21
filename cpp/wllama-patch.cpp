// Patches for upstream code, applied at link time with -Wl,--wrap. See "Patches" in README-dev.md

#include <emscripten/emscripten.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "mtmd.h"

#include "wllama-patch.h"

//
// abort: print the stack trace
//

extern "C" void __real_abort(void);
extern "C" void __wrap_abort(void)
{
  char buf[4096];
  emscripten_get_callstack(EM_LOG_JS_STACK | EM_LOG_NO_PATHS, buf, sizeof(buf));
  for (size_t i = 0; i < sizeof(buf); i++)
  {
    if (buf[i] == '\n')
      buf[i] = '|';
  }
  fprintf(stderr, "@@STACK@@%s\n", buf);
  fflush(stderr);
  __real_abort();
}

//
// ggml threadpool: one shared pool
//

// only one pool for everything (llama, mtmd, graphs with no pool). A second pool needs more workers, and a new worker cannot start while the main thread waits at a barrier
static struct ggml_threadpool *shared_threadpool = nullptr;
static int shared_threadpool_n_threads = 0;
static int shared_threadpool_refs = 0;

extern "C" struct ggml_threadpool *__real_ggml_threadpool_new(struct ggml_threadpool_params *params);
extern "C" struct ggml_threadpool *__wrap_ggml_threadpool_new(struct ggml_threadpool_params *params)
{
  // reuse the pool if it is big enough; a paused pool is a pair with another pool, do not share it
  if (shared_threadpool && !params->paused && params->n_threads <= shared_threadpool_n_threads)
  {
    shared_threadpool_refs++;
    return shared_threadpool;
  }
  struct ggml_threadpool *threadpool = __real_ggml_threadpool_new(params);
  if (!shared_threadpool && threadpool)
  {
    shared_threadpool = threadpool;
    shared_threadpool_n_threads = params->n_threads;
    shared_threadpool_refs = 1;
  }
  return threadpool;
}

extern "C" void __real_ggml_threadpool_free(struct ggml_threadpool *threadpool);
extern "C" void __wrap_ggml_threadpool_free(struct ggml_threadpool *threadpool)
{
  if (threadpool && threadpool == shared_threadpool)
  {
    if (--shared_threadpool_refs > 0)
      return;
    shared_threadpool = nullptr;
    shared_threadpool_n_threads = 0;
  }
  __real_ggml_threadpool_free(threadpool);
}

bool force_single_thread = false;
extern "C" struct ggml_cplan __real_ggml_graph_plan(
    const struct ggml_cgraph *cgraph,
    int n_threads,
    struct ggml_threadpool *threadpool);

extern "C" struct ggml_cplan __wrap_ggml_graph_plan(
    const struct ggml_cgraph *cgraph,
    int n_threads,
    struct ggml_threadpool *threadpool)
{
  return __real_ggml_graph_plan(cgraph, force_single_thread ? 1 : n_threads, threadpool ? threadpool : shared_threadpool);
}

//
// mtmd: record the mtmd context of the server
//

// the mtmd context of the server is private, TTS and realtime need it
mtmd_context *shared_mctx = nullptr;

extern "C" mtmd_context *__real_mtmd_init_from_file(const char *mmproj_fname, const struct llama_model *text_model, const struct mtmd_context_params ctx_params);
extern "C" mtmd_context *__wrap_mtmd_init_from_file(const char *mmproj_fname, const struct llama_model *text_model, const struct mtmd_context_params ctx_params)
{
  shared_mctx = __real_mtmd_init_from_file(mmproj_fname, text_model, ctx_params);
  return shared_mctx;
}

//
// file read: read model files from JS, see "Async file read" in README-dev.md
//

static std::map<FILE *, std::string> s_file_path_map;

namespace wllama_fs
{
  bool ready = false;
  bool use_async = false;

  static const size_t CACHE_SIZE = 1024 * 1024; // 1 MB read-ahead

  std::vector<uint8_t> cache_data;
  off_t cache_start = 0;
  FILE *cache_file = nullptr;

  void make_sure_ready()
  {
    if (ready)
      return;
    use_async = getenv("USE_ASYNC_FILE") != nullptr;
    ready = true;
  }

  size_t try_cache(FILE *f, char *ptr, size_t req_bytes, off_t fpos)
  {
    if (f != cache_file || cache_data.empty())
      return 0;
    if (fpos >= cache_start && fpos + (off_t)req_bytes <= cache_start + (off_t)cache_data.size())
    {
      memcpy(ptr, cache_data.data() + (fpos - cache_start), req_bytes);
      return req_bytes;
    }
    return 0;
  }
}

bool wllama_fs_use_async()
{
  wllama_fs::make_sure_ready();
  return wllama_fs::use_async;
}

// Thin stub - real implementation lives in llama-cpp.js to avoid
// C++ formatter mangling the JS syntax inside EM_ASYNC_JS macros.

// offset, size and out_ptr are passed as double: on wasm32, an i32 above 2GB would arrive in JS as a negative number
EM_ASYNC_JS(size_t, js_file_read_impl, (const char *path_ptr, double offset, double req_size, double out_ptr), {
  return await _wllama_js_file_read(UTF8ToString(Number(path_ptr)), offset, req_size, out_ptr);
});

static size_t js_file_read(const char *path, off_t offset, size_t req_size, void *out_ptr)
{
  return js_file_read_impl(path, (double)offset, (double)req_size, (double)(uintptr_t)out_ptr);
}

extern "C"
{
  FILE *__real_fopen(const char *path, const char *mode);
  int __real_fclose(FILE *f);
  size_t __real_fread(void *ptr, size_t size, size_t nmemb, FILE *f);
  int __real_fseek(FILE *f, long offset, int whence);
  long __real_ftell(FILE *f);

  FILE *__wrap_fopen(const char *path, const char *mode)
  {
    wllama_fs::make_sure_ready();
    FILE *f = __real_fopen(path, mode);
    // JS has the model files only, other files are read from MEMFS
    if (f && (strncmp(path, "models/", 7) == 0 || strncmp(path, "/models/", 8) == 0))
    {
      s_file_path_map[f] = path;
    }
    return f;
  }

  int __wrap_fclose(FILE *f)
  {
    if (wllama_fs::cache_file == f)
    {
      wllama_fs::cache_file = nullptr;
      wllama_fs::cache_data.clear();
    }
    s_file_path_map.erase(f);
    return __real_fclose(f);
  }

  int __wrap_fseek(FILE *f, long offset, int whence)
  {
    return __real_fseek(f, offset, whence);
  }

  long __wrap_ftell(FILE *f)
  {
    return __real_ftell(f);
  }

  size_t __wrap_fread(void *ptr, size_t size, size_t nmemb, FILE *f)
  {
    wllama_fs::make_sure_ready();
    if (!wllama_fs::use_async)
      return __real_fread(ptr, size, nmemb, f);

    auto nit = s_file_path_map.find(f);
    if (nit == s_file_path_map.end())
      return __real_fread(ptr, size, nmemb, f);

    size_t req_bytes = size * nmemb;
    if (req_bytes == 0)
      return 0;

    // ftello and fseeko use 64-bit off_t; ftell and fseek use long, which is 32-bit on wasm32 and fails past 2GB
    off_t fpos = ftello(f);
    if (fpos < 0)
      return 0;

    // Large reads (>= 1 MB): write directly into ptr, skip cache entirely.
    if (req_bytes >= wllama_fs::CACHE_SIZE)
    {
      size_t actual = (size_t)js_file_read(
          nit->second.c_str(), fpos, req_bytes, ptr);
      if (actual == 0)
        return 0;
      size_t copy_bytes = std::min(req_bytes, actual);
      fseeko(f, fpos + (off_t)copy_bytes, SEEK_SET);
      return copy_bytes / size;
    }

    // Small reads: try cache first.
    size_t cached = wllama_fs::try_cache(f, (char *)ptr, req_bytes, fpos);
    if (cached == req_bytes)
    {
      fseeko(f, fpos + (off_t)req_bytes, SEEK_SET);
      return nmemb;
    }

    // Cache miss: fetch a full CACHE_SIZE block from main thread.
    wllama_fs::cache_data.resize(wllama_fs::CACHE_SIZE);
    size_t actual = (size_t)js_file_read(
        nit->second.c_str(), fpos, wllama_fs::CACHE_SIZE,
        wllama_fs::cache_data.data());

    wllama_fs::cache_data.resize(actual);
    wllama_fs::cache_file  = f;
    wllama_fs::cache_start = fpos;

    if (actual == 0)
      return 0;

    size_t copy_bytes = std::min(req_bytes, actual);
    memcpy(ptr, wllama_fs::cache_data.data(), copy_bytes);
    fseeko(f, fpos + (off_t)copy_bytes, SEEK_SET);

    return copy_bytes / size;
  }
}

//
// model load: upload GPU weights in small chunks
//
// Without mmap, llama.cpp copies each GPU tensor through a CPU buffer as big as the tensor (417 MB for the token embedding of a 2B model).
// The wasm heap never shrinks, so this peak stays as page memory after the load.
// The loader has a path with 4 small buffers, but it needs the async, events and host buffer caps, which ggml-webgpu does not report although it implements them.
// So we report them while a model loads, and only then: llama_context reads the same caps to turn on pipeline parallelism.
//

static bool loading_model = false;

static bool is_webgpu(ggml_backend_dev_t dev)
{
  return dev && strcmp(ggml_backend_dev_name(dev), "WebGPU") == 0;
}

// rpc workers (see wllama-rpc-transport.cpp), their devices are named "RPC<n>[<endpoint>]"
static bool is_rpc(ggml_backend_dev_t dev)
{
  return dev && strncmp(ggml_backend_dev_name(dev), "RPC", 3) == 0;
}

extern "C" struct llama_model *__real_llama_model_load_from_file(const char *path_model, struct llama_model_params params);
extern "C" struct llama_model *__wrap_llama_model_load_from_file(const char *path_model, struct llama_model_params params)
{
  loading_model = true;
  struct llama_model *model = __real_llama_model_load_from_file(path_model, params);
  loading_model = false;
  return model;
}

extern "C" struct llama_model *__real_llama_model_load_from_splits(const char **paths, size_t n_paths, struct llama_model_params params);
extern "C" struct llama_model *__wrap_llama_model_load_from_splits(const char **paths, size_t n_paths, struct llama_model_params params)
{
  loading_model = true;
  struct llama_model *model = __real_llama_model_load_from_splits(paths, n_paths, params);
  loading_model = false;
  return model;
}

extern "C" void __real_ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props *props);
extern "C" void __wrap_ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props *props)
{
  __real_ggml_backend_dev_get_props(device, props);
  if (loading_model && (is_webgpu(device) || is_rpc(device)))
  {
    props->caps.async = true;
    props->caps.events = true;
    props->caps.host_buffer = true;
  }
}

extern "C" ggml_backend_buffer_type_t __real_ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
extern "C" ggml_backend_buffer_type_t __wrap_ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device)
{
  ggml_backend_buffer_type_t buft = __real_ggml_backend_dev_host_buffer_type(device);
  if (!buft && loading_model && (is_webgpu(device) || is_rpc(device)))
  {
    // WriteBuffer() (WebGPU) and set_tensor_async (ggml-rpc) copy the data at call time, so plain CPU memory works as a staging buffer
    return ggml_backend_cpu_buffer_type();
  }
  return buft;
}

//
// WebGPU: skip queue waits when the queue is idle
//
// ggml-webgpu waits for the queue on each ggml_backend_synchronize(), about 15 times per token, even when nothing was submitted since the last wait.
// Each wait is slow on the web (100 ms on Firefox, see src/patch/webgpu-poll.js), so we return at once when all submits are already complete.
// Assumes one queue, used only from the thread that created it (true for emdawnwebgpu).
//

#ifdef WLLAMA_PATCH_WEBGPU

#include <webgpu/webgpu.h>

extern "C"
{
  void __real_wgpuQueueSubmit(WGPUQueue queue, size_t count, const WGPUCommandBuffer *commands);
  WGPUFuture __real_wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo info);
  WGPUFuture __real_wgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo info);
  WGPUWaitStatus __real_wgpuInstanceWaitAny(WGPUInstance instance, size_t count, WGPUFutureWaitInfo *futures, uint64_t timeout_ns);
  WGPUStatus __real_wgpuAdapterGetLimits(WGPUAdapter adapter, WGPULimits *limits);
}

static uint64_t n_submitted = 0;
static uint64_t n_completed = 0; // submits [1, n_completed] are complete

// future returned for a skipped wait, it is always complete
static constexpr uint64_t COMPLETED_FUTURE_ID = UINT64_MAX;

// wraps a callback, to record which submits are complete when it runs
template <typename callback_t, typename status_t, status_t success>
struct wllama_wgpu_callback
{
  callback_t callback;
  void *userdata1;
  void *userdata2;
  uint64_t n_submitted_at_call;

  static void call(status_t status, WGPUStringView message, void *self_ptr, void *)
  {
    auto *self = (wllama_wgpu_callback *)self_ptr;
    if (status == success)
    {
      n_completed = std::max(n_completed, self->n_submitted_at_call);
    }
    self->callback(status, message, self->userdata1, self->userdata2);
    delete self;
  }

  template <typename info_t>
  static info_t wrap(info_t info)
  {
    if (info.callback)
    {
      info.userdata1 = new wllama_wgpu_callback{info.callback, info.userdata1, info.userdata2, n_submitted};
      info.userdata2 = nullptr;
      info.callback = call;
    }
    return info;
  }
};

using work_done_callback = wllama_wgpu_callback<WGPUQueueWorkDoneCallback, WGPUQueueWorkDoneStatus, WGPUQueueWorkDoneStatus_Success>;
// ggml maps a staging buffer right after it submits the copy to it, so a completed map means all earlier submits are complete
using map_callback = wllama_wgpu_callback<WGPUBufferMapCallback, WGPUMapAsyncStatus, WGPUMapAsyncStatus_Success>;

extern "C" void __wrap_wgpuQueueSubmit(WGPUQueue queue, size_t count, const WGPUCommandBuffer *commands)
{
  n_submitted++;
  __real_wgpuQueueSubmit(queue, count, commands);
}

extern "C" WGPUFuture __wrap_wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo info)
{
  // WriteBuffer() is not counted: it copies the data at call time and the queue keeps the order
  if (n_completed == n_submitted && info.mode == WGPUCallbackMode_AllowSpontaneous)
  {
    if (info.callback)
    {
      info.callback(WGPUQueueWorkDoneStatus_Success, WGPUStringView{nullptr, 0}, info.userdata1, info.userdata2);
    }
    return WGPUFuture{COMPLETED_FUTURE_ID};
  }
  return __real_wgpuQueueOnSubmittedWorkDone(queue, work_done_callback::wrap(info));
}

extern "C" WGPUFuture __wrap_wgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo info)
{
  return __real_wgpuBufferMapAsync(buffer, mode, offset, size, map_callback::wrap(info));
}

extern "C" WGPUWaitStatus __wrap_wgpuInstanceWaitAny(WGPUInstance instance, size_t count, WGPUFutureWaitInfo *futures, uint64_t timeout_ns)
{
  // emdawnwebgpu rejects unknown futures, so do not pass COMPLETED_FUTURE_ID to it
  bool any_completed = false;
  for (size_t i = 0; i < count; i++)
  {
    if (futures[i].future.id == COMPLETED_FUTURE_ID)
    {
      futures[i].completed = true;
      any_completed = true;
    }
  }
  if (any_completed)
  {
    return WGPUWaitStatus_Success;
  }
  return __real_wgpuInstanceWaitAny(instance, count, futures, timeout_ns);
}

// ggml adds the tensor sizes of one buffer in size_t, and only starts a new buffer above maxBufferSize
// on wasm32, a limit near 4GB lets this sum wrap around, so a model with more than 4GB on the GPU gets a buffer that is too small
// 2GB keeps the sum of a buffer and the next tensor under 4GB; ggml then uses more buffers
static const uint64_t WGPU_MAX_BUFFER_SIZE_32 = 0x7fff0000;
extern "C" WGPUStatus __wrap_wgpuAdapterGetLimits(WGPUAdapter adapter, WGPULimits *limits)
{
  WGPUStatus status = __real_wgpuAdapterGetLimits(adapter, limits);
  if (status == WGPUStatus_Success && sizeof(size_t) < sizeof(uint64_t))
  {
    limits->maxBufferSize = std::min(limits->maxBufferSize, WGPU_MAX_BUFFER_SIZE_32);
    limits->maxStorageBufferBindingSize = std::min(limits->maxStorageBufferBindingSize, WGPU_MAX_BUFFER_SIZE_32);
  }
  return status;
}

#endif // WLLAMA_PATCH_WEBGPU
