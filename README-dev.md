# wllama

Wllama is a webassembly binding of llama.cpp. It contains the main source code of llama.cpp compiled to wasm (with emscripten), plus a wrapper to provide various convenient APIs, including: downloading and caching models, compatibility, etc.

## Project structure

The project has these directories:
- `src`: the main typescript source code
- `cpp`: C++ interface (`cpp/native-test`: a native harness for it, see "Debugging the C++ glue natively")
- `scripts`: various scripts for development
- `examples`: various examples

The project has these main components:
- `wllama.ts`: the main public API
- `model-manager.ts`: relies on cache manager to manage models. For example, a model can be composed of multiple files
    - `cache-manager.ts`: interface for managing cache files. It uses OPFS under the hood
    - `huggingface.ts`: utility for managing models downloading from hugging face hub
- `worker.ts`: the worker manager that will be responsible of starting the emscripten worker and maintaining the communication with it
- `glue.ts`: GLUE implementation
- `wllama.cpp`: the main C++ interface

### GLUE

GLUE is a home-grown binary protocol inspired by Protobuf. It is used internally to communicate between the wasm context and the JavaScript context of wllama.

The main goal of GLUE is to allow a type-safe interface with low overhead. It works by serializing messages into `ArrayBuffer` and transferring them using [Transferable objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects), which avoids copying.

**Wire format:**
- 4 bytes - magic number (`GLUE`)
- 4 bytes - version number (`GLUE_VERSION`)
- 8 bytes - message prototype ID
- 4 bytes - message length (unsigned)
- message fields, each encoded as:
  - 4 bytes data type (e.g. `int`, `float`, `str`, `raw`, and array variants)
  - 4 bytes size (only for arrays and strings)
  - data bytes

**Supported field types:** `str`, `int`, `float`, `bool`, `raw` (arbitrary bytes), and array variants of each.

Upon build, `generate_glue_prototype.js` reads `glue.hpp` and generates `glue/messages.ts`, which provides the TypeScript-side message types used throughout the codebase.

### Threading model

Wllama ships a **single wasm build** that supports both single-threaded and multi-threaded execution. The number of threads is determined at runtime rather than at compile time.

At startup, wllama checks whether the browser supports `SharedArrayBuffer` (required for wasm threads). This check validates both the existence of `SharedArrayBuffer` and whether the wasm atomics feature is available (COOP/COEP headers must be set by the server for `SharedArrayBuffer` to be accessible).

The thread pool size is passed to emscripten via `-sPTHREAD_POOL_SIZE=Module["pthreadPoolSize"]`:
- If the browser **supports** shared memory: `pthreadPoolSize` is set to the desired thread count (defaults to `hardwareConcurrency / 2`)
- If the browser **does not support** shared memory: `pthreadPoolSize` is set to `0`, which disables pthreads entirely and falls back to single-threaded execution

This logic lives in `wllama.ts` (`isSupportMultiThread()` from `utils.ts` performs the feature detection).

**The maximum number of threads is fixed by the number of threads of the ggml threadpool (`n_threads`). Do NOT increase `pthreadPoolSize` above it, in any case.** If some code needs more threads than that, a new worker cannot start while the wasm thread waits, so it hangs. Fix that code instead: make it use the ggml threadpool (see `shared_threadpool` in `cpp/wllama-patch.cpp`) or no extra threads. Growing the pool (for example `+ 4` or `* 2`) is not an accepted fix.

The only exception is the rpc workers (see "RPC workers" below): ggml-rpc does the I/O of each rpc server in its own thread, which cannot use the ggml threadpool, so the pool of the main wasm has `n_threads` plus one thread per rpc worker. This number is fixed when the module starts, it is not a workaround for unexpected threads.

### Realtime audio

The realtime session of llama-server needs a thread and blocking reads, but wllama has one thread and polls. So `wllama-context.h` includes `server-realtime.cpp` directly (it is not in the CMake source list) and calls the `handle()` and `on_result()` steps of `realtime_conversation` itself, without `run()`:
- `realtime_send` passes a client event to `handle()`. `server_response_reader::next()` runs the queue while it waits for the result of a control
- `get_result` runs one step of the loop, passes the result to `on_result()` and returns the events
- `cancel` ends the session

If upstream changes `realtime_conversation`, the fix belongs in `realtime_session_conversation`. Live ASR has no such class upstream (it is inside the `ws_realtime` route), so `realtime_session_live` is a copy of that route.

## Startup process

Upon startup, these steps are performed:
- `ProxyToWorker` is created in the main wllama JS context
- A web worker is spawned, the code is taken from `workers-code/generated.ts`
- The worker loads emscripten code, sets up the environment then eventually calls the `main()` inside `wllama.cpp`. These preparation steps are injected (see `llama-cpp.js`):
    - Hooking `printf` functions
    - Setting up HeapFS
    - Setting up communication callbacks

## File access

Wllama employs some tricks to avoid making copies while reading GGUF files. The runtime uses one of these 2 mechanisms. See `workers-code/llama-cpp.js` for the implementation.

Please note that wllama only accepts `Blob` as input data.

### Async file read

This implementation hooks into `fopen`, `fseek` and `fread`, and forwards these calls to the main thread (via message port), where we eventually call `Blob.slice()` to read the data. Because of the asynchronous execution via `onmessage` and `postMessage`, JSPI / Asyncify is required.

Upon running, action `fs.alloc` is fired to indicate that the file can be read through JSPI / Asyncify call. The actual buffer won't be allocated for the file, but only the metadata is.

When wasm calls `fread()`:
- `fread()` calls `await fileRead()` in the JS context
- `fileRead()` posts a message of type `fs.read_req` to the main thread
- Main thread uses `Blob.slice()` to read the data, then sends it back via a `fs.read_res` message
- Worker's `onmessage` receives the message and resumes the awaiting coroutine

Note:
- While awaiting the read data, the worker should not have any other activities (a global variable is used as a guard and will raise an exception on any incoming messages)
- The minimum read size is 1MB. If less than this amount is requested, the full 1MB block is cached for subsequent reads. This is because reading GGUF metadata frequently involves reads of less than 1KB at a time, which can become a bottleneck without caching.
- On the JS side (`_wllama_js_file_read` in `llama-cpp.js`), the main thread is asked for 32MB at most per read: a bigger read is split in 32MB chunks. Each chunk is a new `ArrayBuffer` that is garbage once copied to the heap, so bigger chunks pile up more garbage while loading. The weights for the GPU and rpc workers are read 1MB at a time, one round trip each, so a read that follows the previous one gets a read-ahead: 8 times the read, then 4 times more each time, up to 32MB, kept for the next reads. A read somewhere else (the loader jumps between tensors) resets it and gets only what it asks for.
- Files from the OPFS cache (`loadModelFromUrl`, `ModelManager`) are read with a sync access handle instead (`opfsRead` in `llama-cpp.js`): straight into the wasm heap, in the worker, without a `Blob` or a round trip to the main thread. `storage/opfs.ts` remembers the OPFS path of each `File` it returns, `worker.ts` sends it with `fs.alloc`, and the worker opens the file itself: Safari cannot send a `FileSystemFileHandle` to a worker. The handle is opened in `read-only` mode when needed and closed after each wasm call, so the cache can change or delete the file between calls. If it cannot be opened (another context has the file open, no sync access handles, or the file changed size), the file is read from its `Blob` as before. On a Galaxy Tab S10 FE, reads go from about 180MB/s to 1-4GB/s: LFM2.5-1.2B Q4_K_M (731MB) loads in 1.3-2.0s instead of 4.7-5.1s. On an iPhone SE 3 (Safari 27.2, compat build), LFM2-350M Q4_K_M (229MB) loads in 1.9-2.1s instead of 2.5-2.7s. llama.cpp reads the tensors of the CPU in the order the model creates them, not in file order, so a bigger read-ahead does not help the `Blob` path (#300)
- Env var `USE_ASYNC_FILE` is used to signal from JS to wasm that we are using async file read (upon starting the module). If `USE_ASYNC_FILE` is not set, we fallback to HeapFS/mmap case (see in next section)

Memory use while loading:
- The reads are pulled by llama.cpp: the main thread only reads the range that `fread()` asks for, it does not stream the whole file
- The main thread transfers the `ArrayBuffer` to the worker (no copy), then the worker copies it to the destination in the wasm heap. This is the only copy
- So the whole file is never in the wasm heap at once (unlike HeapFS below)
- The wasm heap never shrinks: memory freed after the load goes back to malloc, not to the browser. So the peak during the load stays as page memory, keep it small
- Tensors on the CPU stay in the wasm heap. Tensors on the GPU or on rpc workers are copied through 4 buffers of 1MB (see the `ggml_backend_dev_get_props` patch in "Patches"), then uploaded, so they do not stay in the wasm heap
- With all layers on the GPU, the token embedding is also put on the GPU (load option `token_embd_on_gpu`), otherwise llama.cpp keeps a CPU copy for the input lookup

### HeapFS

HeapFS is a lightweight wrapper around emscripten's default FS driver. The main goal is to allow `mmap()` to map to existing data instead of copying it (the default emscripten behavior).

These steps are performed:

- Action `fs.alloc` is fired to create the file handle and file buffer in the wasm context
- The main thread then creates and holds a `ReadableStream` for the `Blob`
- The main thread reads the file chunk by chunk, streaming it to the worker via `fs.write` messages
- Once streaming is finished, the `ReadableStream` is closed
- The model load is then triggered with `mmap = true`, and `mmap()` is wrapped to return a pointer to the correct data in the buffer allocated in step 1

The main downside of this approach is that on WebGPU, even though some tensors can be offloaded to the GPU, we still need to allocate the full model in main memory. For example, a 4GB model will still occupy 4GB of main memory, even if half of the layers (~2GB) are offloaded to the GPU.

## RPC workers

Both builds use 32-bit memory, so one wasm instance has 4GB at most. Without WebGPU, a bigger model is split across several wasm instances: the main one, plus `rpc_workers` workers that each run the same wasm as a ggml-rpc server (`wllama_rpc_serve`). llama.cpp sees each worker as an RPC device (same as `--rpc`) and puts layers on it.

- By default, `pickRpcWorkers()` in `src/wllama.ts` uses workers only when WebGPU is not used and the model does not fit in one instance. `countRpcWorkers()` in `src/gguf-split.ts` estimates the number from the GGUF header: the main instance keeps the input embeddings, the workers get the layers and the output. `token_embd` goes to the first worker, as `keep_tok_embd_on_gpu()` picks the first GPU-type device, and rpc devices are GPU-type. Set `rpc_workers` to force a number, or 0 to disable.
- `cpp/wllama-rpc-transport.cpp` replaces `transport.cpp` of ggml-rpc (see `CMakeLists.txt`). The bytes go through 2 SharedArrayBuffer rings per worker, and the reader waits with `Atomics.wait`. The JS side is `src/workers-code/rpc-transport.js`.
- On the main wasm, ggml-rpc does its I/O in its own thread (one per worker), so the transport also runs in the pthread workers: `worker.ts` adds it to their script, and the channels are posted to each pthread worker before emscripten starts it. The pthread pool gets one more thread per worker.
- Multi-thread is required.
- ggml-rpc has 2 bugs on 32-bit hosts (#340), the transport works around them on the wire: message sizes are widened from `size_t` to `uint64_t`, and the high half of the node ids of `GRAPH_COMPUTE` is cleared.

## Packing the wasm

After each build, `scripts/pack_wasm.js` packs the emscripten output into a single `wllama.wasm`. It adds 2 custom sections:

- `wllama_js`: the emscripten JS glue (`build/wllama.js`). At runtime, `src/worker.ts` compiles the wasm, reads the JS from this section and runs it in the worker. So the JS and the wasm of one build always ship together, and `wllama.js` is not tracked in git.
- `name`: function names for stack traces, see below.

### Function names in stack traces

The release wasm has no debug info, so `scripts/pack_wasm.js` adds a standard wasm `name` section with cleaned names after each build. Browsers read this section themselves, so a stack trace shows `server_response::send` without any decoding in JS.

Emscripten's `--emit-symbol-map` flag produces a `.js.symbols` file mapping each wasm function index to its demangled C++ name. The script reads this file from the build directory and writes the result to `src/wasm/` or `compat/wasm/`. It skips a section that is already there, so running it twice does not add a second one.

The script runs as part of the docker build (see `scripts/docker-compose.yml`). It can also be run by hand:

```sh
node scripts/pack_wasm.js build/wllama.wasm src/wasm/wllama.wasm
```

#### Name cleaning rules

Raw demangled names can be hundreds of characters. The following rules are applied in order:

1. **std:: collapse** - any name starting with `std::` is replaced with the single hint `std::...`
2. **Lambda/closure extraction** - names containing `::$_N` or `::'lambda'` are replaced with the nearest enclosing context (the segment inside the last `<...>` before the marker)
3. **Parameter stripping** - parameter lists are dropped; empty `()` is kept, non-empty is removed entirely
4. **libc++ internals** - `::__1::`, `::__2::`, etc. are collapsed to `::`
5. **ABI tags** - `[abi:...]` annotations are removed
6. **Template truncation** - template argument content longer than 10 characters is truncated to `<first10chars...>`
7. **Final cleanup** - double `::::` collapsed, whitespace normalised

## Version check

`LIBLLAMA_VERSION` exists on both sides: `scripts/build_worker.sh` writes it to `src/workers-code/generated.ts`, and `CMakeLists.txt` compiles it into the wasm. `wllama_start()` returns the wasm version, and `Wllama` raises a `version_mismatch` error before it loads the model if the two commits differ. Set `WllamaConfig.skipVersionCheck` to get a warning instead.

Only the commit is compared, because the build number is wrong in a shallow clone.

`scripts/check_wasm_version.js` does the same check on the wasm files without running them. CI runs it before `npm run build`, and `scripts/check_package_size.js` runs it before publish. Publish also requires the compat wasm, so a build with `SKIP_COMPAT=1` cannot be published.

## Patches

We do not edit `llama.cpp/`. When upstream code does not fit the web, we patch it from our side:
- `cpp/wllama-patch.cpp`: C++ patches. They replace upstream calls at link time with `-Wl,--wrap` (see `CMakeLists.txt`). `--wrap` only changes calls between object files, so wrap a function that the upstream code calls from another file (for example the emdawnwebgpu C API), not a function in the same file as its callers. `cpp/wllama-patch.h` declares the state that the rest of the glue uses.
- `src/patch/`: JS patches. `scripts/build_worker.sh` embeds them in `src/workers-code/generated.ts`, and `src/worker.ts` runs them in the worker before the wasm module starts.
- `scripts/binaryen-asyncify-try.patch`: a binaryen patch, see "Asyncify with wasm exceptions" below.

The C++ patches:
- `abort`: print the stack trace before abort
- `ggml_threadpool_new`, `ggml_threadpool_free`, `ggml_graph_plan`: one shared ggml threadpool, see "Threading model"
- `mtmd_init_from_file`: record the mtmd context of the server (`shared_mctx`), TTS and realtime need it
- `fopen`, `fclose`, `fread`, `fseek`, `ftell`: read model files from JS, see "Async file read"
- `llama_model_load_from_file`, `llama_model_load_from_splits`, `ggml_backend_dev_get_props`, `ggml_backend_dev_host_buffer_type`: while a model loads, report the async, events and host buffer caps for WebGPU and rpc devices, so llama.cpp uploads the weights through 4 small buffers instead of one buffer as big as the biggest tensor. The wasm heap never shrinks, so that big buffer stays as page memory after the load
- `wgpuQueueSubmit`, `wgpuQueueOnSubmittedWorkDone`, `wgpuBufferMapAsync`, `wgpuInstanceWaitAny`: see "WebGPU queue waits" below
- `wgpuAdapterGetLimits`: cap `maxBufferSize` at 2GB. ggml adds the tensor sizes of one buffer in `size_t`, so with a limit near 4GB the sum wraps around on wasm32, and a model with more than 4GB on the GPU fails to load

Remove a patch when upstream fixes the problem.

### WebGPU queue waits

ggml-webgpu waits for the GPU queue in `ggml_backend_synchronize()`. llama.cpp calls it about 15 times per decoded token: before each graph input upload, before the logits read back, after decode and several times in the sampler. Only the logits read back (one `mapAsync`) is necessary. Each wait goes through emdawnwebgpu to a JS promise, so it is slow on the web, and very slow on Firefox (see below).

`cpp/wllama-patch.cpp` wraps 4 emdawnwebgpu functions:
- `wgpuQueueSubmit` counts the submits
- `wgpuQueueOnSubmittedWorkDone` and `wgpuBufferMapAsync` wrap the callback, to record which submits are complete. A completed map counts too, because ggml maps a staging buffer right after it submits the copy to it
- If all submits are complete, `wgpuQueueOnSubmittedWorkDone` calls the callback at once and returns a special future. `wgpuInstanceWaitAny` returns at once for this future (emdawnwebgpu rejects unknown futures)

`WriteBuffer()` is not counted: WebGPU copies the data at call time and the queue keeps the order.

### Firefox WebGPU poll

Firefox resolves WebGPU promises only when it polls the device, and it polls on a 100 ms timer ([bug 1870699](https://bugzilla.mozilla.org/show_bug.cgi?id=1870699)). So each wait costs about 100 ms, and a model decodes at less than 1 token/s.

Firefox also polls the device (and waits for it) when script reads back a WebGPU canvas (`GetFrontBufferSnapshot` in `dom/webgpu/ipc/WebGPUParent.cpp`). The poll runs all completed callbacks, ours too. `src/patch/webgpu-poll.js` uses this: on Firefox, after each `onSubmittedWorkDone()` and `mapAsync()`, it clears a 1x1 WebGPU canvas on the same device and draws it into a 2D canvas. The canvas needs new work for each read back, otherwise Firefox gives the last frame without a poll. If the patch fails, it disables itself and the waits use the 100 ms timer again.

Performance numbers of the 2 patches: https://github.com/ngxson/wllama/pull/341

### Asyncify with wasm exceptions

The default build uses JSPI and relaxed SIMD. If the browser misses one of them, wllama uses the compat build (`needCompat()` in `src/utils.ts`): asyncify, no relaxed SIMD. Both builds use 32-bit memory.

The compat build uses asyncify and wasm exceptions (`-fwasm-exceptions`). Upstream asyncify stops with `UNREACHABLE` when a call that can suspend is inside a `try` ([binaryen#4470](https://github.com/WebAssembly/binaryen/issues/4470)). C++ puts most calls inside a `try` (to run the destructors), so without the patch, the compat build must use JS exceptions (`-fexceptions`), which are slower and bigger.

The patch adds `try` support to asyncify:
- Rewind goes into the `try` body again, so the exception handlers are back in place
- A rewind never throws, so it never goes into a `catch` body. If code tries to suspend inside a `catch` body, it traps
- Only the legacy exception format is supported, not exnref (`try_table`)

The docker image builds `wasm-opt` with the patch, at the binaryen commit of the emsdk version, then puts it in place of the emsdk `wasm-opt` (see `scripts/docker-compose.yml`). The image tag has the hash of the patch, so a new patch makes a new image. The patch does not change the JSPI build, so with `SKIP_COMPAT=1` (CI and the upstream sync) the image does not build binaryen, which saves several minutes.

The patched `wasm-opt` is also copied to `scripts/.binaryen-cache` (git-ignored), named by the image tag. The next image build uses it instead of building binaryen again. CI caches this folder.

To test the compat build: build it (without `SKIP_COMPAT`), then `npm run test:compat`. It runs `src/testing/wllama.test.ts` on `compat/wasm/wllama.wasm`, with JSPI hidden on the page. Use `SKIP_MAIN=1` to build only compat. CI runs it in the `test-compat` job, in parallel with the main tests.

To test a model bigger than 4GB: `npm run test:big`. It loads a 6GB model made on the fly (`src/testing/fake-gguf.ts`, no download) as CPU only (rpc workers), GPU only, and about 5GB on CPU with the rest on GPU, which is expected to fail for now. It needs WebGPU and about 12GB of free RAM, so CI does not run it.

## Debugging backend ops

> [!IMPORTANT]
>
> By default, the build does NOT include `test-backend-ops` to save space. If you need to run it, please clone the repo and build it yourself, instructions below

Requirements:
- You have Docker installed and running on your machine
- On Windows, please use WSL

1. Clone this repo locally: `git clone --recurse-submodules https://github.com/ngxson/wllama.git`
2. `npm run build:test && npm run build`
3. `npm run serve` and open http://localhost:8080/examples/test-backend-ops/

Note: A debugging build cannot be merged to `master` or publish to npm

## Debugging the C++ glue natively

`cpp/native-test/` contains a native (non-wasm) harness that compiles `cpp/wllama-context.h` together with llama.cpp's server sources and drives it exactly like the JS worker does: single-threaded, one `run_loop()` iteration per `get_result` poll, glue-encoded messages, per-request response readers.

### Why

Debugging the C++ glue through the emscripten/browser stack is slow and mostly blind. This harness runs the same code natively, so you get fast rebuilds, `printf`/`lldb`, and clean stderr ordering.

It is also a **bisection tool** for wasm-only failures:

- Bug reproduces here -> it is in `cpp/wllama-context.h` or in llama.cpp's server code. Debug it natively.
- Bug only happens in the browser -> suspect the wasm environment instead: the WebGPU backend, JSPI/asyncify, emscripten build flags, worker interleaving. (Example: the ggml-webgpu multi-output bug from issue #261 passed here but crashed in wasm.)

### Build

```sh
cmake -S cpp/native-test -B cpp/native-test/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build cpp/native-test/build --target native-test -j
```

### Run

Any small GGUF works; the test suite's tiny model is a good default:

```sh
curl -sL -o /tmp/stories15M-q4_0.gguf \
  "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf"
cpp/native-test/build/native-test /tmp/stories15M-q4_0.gguf
```

Expected output ends with `== ALL OK`.

### Adapting it

`main.cpp` is meant to be edited per investigation:

- The default scenario posts request A, polls it 4 times (mid-generation), then posts B and C - forcing mixed generation+prompt batches and multi-slot decoding. Change the interleaving to reproduce other patterns.
- `call_action()` + `set_all_null()` mimic the JS glue encoder: null every field first, then set `value` **and** `dtype` for each field you need (fields default to "present", and e.g. an empty `pooling_type` string would be parsed and rejected).
- The stubs at the top replace emscripten-only symbols and the build-info / `common_log` functions that the wasm link resolves differently; if the linker reports a new undefined symbol after a llama.cpp sync, add a stub for it there.
- Keep the source lists in `CMakeLists.txt` in sync with the root `CMakeLists.txt` when the wasm build's file lists change.
- `cpp/wllama-patch.cpp` (the `-Wl,--wrap` patches of the wasm link) is not built here: the macOS linker does not have `--wrap`. `main.cpp` defines the state that `cpp/wllama-patch.h` declares. TTS and realtime need the mtmd context that `__wrap_mtmd_init_from_file` records, so they do not work in this harness.

## Testing on a phone or tablet

`scripts/dev-device/` is a dev server to run wllama on an Android phone or tablet (or any browser), driven by an agent or a script on this machine. The device opens a page that stays connected and keeps the screen awake. The page under test runs in an iframe of it, so it can be reloaded without losing the connection. The agent sends commands, the device runs them in the iframe and sends back the result and the console output (wllama's logs included).

```sh
npm run dev:device                                      # serves the repo on port 8090 (HTTP) and 8091 (HTTPS)
npm run dev:device -- --models ./path/to/models          # also serves a folder of GGUF files at /models/ (Range requests work)
npm run dev:device -- --root ../other-worktree           # serves another checkout
```

Then open the device page on the device. The page must be a secure context, or the browser disables SharedArrayBuffer (no multi-thread), the Wake Lock API and OPFS (the model cache):

- With adb (USB, or wireless debugging), recommended: `npm run dev:device -- --adb` forwards the port (`adb reverse tcp:8090 tcp:8090`), keeps the screen on while the device is plugged in (`adb shell svc power stayon usb`, turned off again when the server stops) and opens `http://localhost:8090/device` in Chrome on the device. `localhost` is a secure context
- Without adb, on the same Wi-Fi: open one of the `https://<ip>:8091/device` URLs that the server prints, and accept the certificate warning once. The server makes a self-signed certificate for its LAN addresses (`scripts/dev-device/.certs`, git-ignored)

Keeping the device awake: the page holds a screen wake lock (the "awake" field shows `yes (wake lock)`) and takes it again when the browser releases it. The browser can still release it during a long run (seen on a Galaxy Tab S10 FE), and then the device sleeps and locks: with a device on adb, use `--adb`, which keeps the screen on from the system side. When the wake lock is refused (Safari only gives it after a tap on the page) or missing, the page plays a hidden muted video instead (`video`), which needs no tap: an iPhone SE 3 (Safari 27.2) stayed awake and connected for over an hour with it.

An iPhone has no `adb reverse`: open the `https://<ip>:8091/device` URL on it. With Xcode and a paired iPhone, this opens it from this machine: `xcrun devicectl device process launch --device <id from xcrun devicectl list devices> --payload-url https://<ip>:8091/device com.apple.mobilesafari`. Safari on iOS runs the compat build: call `setCompat({ wasm: '/compat/wasm/wllama.wasm' })` in the code under test, or wllama fetches the published compat build from the CDN, which can be another version.

Commands, they print JSON and exit with a non-zero code on failure:

```sh
node scripts/dev-device/index.mjs status                 # connected devices, with their user agent, cores, WebGPU adapter, isolation, storage quota
node scripts/dev-device/index.mjs open /examples/basic/  # load a page in the iframe (default: scripts/dev-device/blank.html)
node scripts/dev-device/index.mjs reload [--hard]        # reload the page under test (--hard: the whole device page)
node scripts/dev-device/index.mjs eval "document.title"  # run an expression or a function body (with return) in the page under test, it can await
node scripts/dev-device/index.mjs run ./test.js          # same, with the code of a file
node scripts/dev-device/index.mjs logs [--follow]        # console output of the devices
```

`--timeout <seconds>` (default 120) and `--device <id>` work with the commands. Without `--device`, a command goes to a device whose page is visible (the browser throttles hidden tabs), then to the one that connected last. A file that loads a model and returns the timings:

```js
const { Wllama } = await import('/esm/index.js'); // run `npm run build:wasm` then `npm run build` first
const wllama = new Wllama({ default: '/src/wasm/wllama.wasm' });
const t0 = performance.now();
await wllama.loadModelFromUrl(location.origin + '/models/model.gguf', { n_threads: 4 });
const loadMs = performance.now() - t0;
const res = await wllama.createCompletion({ prompt: 'Once upon a time,', max_tokens: 16, temperature: 0 });
await wllama.exit();
return { loadMs, text: res.choices[0].text };
```

When a served file changes (`esm/` after `npm run build`, `examples/`, `src/wasm/`...), the server reloads the page under test on the devices. Changes of `scripts/dev-device/device.html` reload the whole device page; changes of `scripts/dev-device/index.mjs` need a restart of the server. `--no-watch` turns this off.

The commands are only accepted from this machine (`/api/cmd`, `/api/status` and `/api/logs` answer 403 to other addresses), but the files and the device page are served to the whole network: only run it on a network you trust.

## Build process

The build process uses emscripten in docker to compile the project.

After compilation, `generate_glue_prototype.js` is called to generate the GLUE message types to be used in TypeScript.

Built wasm file will then be copied to the `src` directory.

Finally, `build_worker.sh` is called to generate the web worker code.
