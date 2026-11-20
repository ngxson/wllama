// Patches for upstream code, applied at link time with -Wl,--wrap. See "Patches" in README-dev.md

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cstdint>

//
// WebGPU: skip queue waits when the queue is idle
//
// ggml-webgpu waits for the queue on each ggml_backend_synchronize(), about 15 times per token, even when nothing was submitted since the last wait.
// Each wait is slow on the web (100 ms on Firefox, see src/patch/webgpu-poll.js), so we return at once when all submits are already complete.
// Assumes one queue, used only from the thread that created it (true for emdawnwebgpu).
//

extern "C"
{
  void __real_wgpuQueueSubmit(WGPUQueue queue, size_t count, const WGPUCommandBuffer *commands);
  WGPUFuture __real_wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo info);
  WGPUFuture __real_wgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo info);
  WGPUWaitStatus __real_wgpuInstanceWaitAny(WGPUInstance instance, size_t count, WGPUFutureWaitInfo *futures, uint64_t timeout_ns);
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
