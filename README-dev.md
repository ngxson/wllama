# wllama

Wllama is a webassembly binding of llama.cpp. It contains the main source code of llama.cpp compiled to wasm (with emscripten), plus a wrapper to provide various convenient APIs, including: downloading and caching models, compatibility, etc.

## Project structure

The project has these directories:
- `src`: the main typescript source code
- `cpp`: C++ interface
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
- Env var `USE_ASYNC_FILE` is used to signal from JS to wasm that we are using async file read (upon starting the module). If `USE_ASYNC_FILE` is not set, we fallback to HeapFS/mmap case (see in next section)

### HeapFS

HeapFS is a lightweight wrapper around emscripten's default FS driver. The main goal is to allow `mmap()` to map to existing data instead of copying it (the default emscripten behavior).

These steps are performed:

- Action `fs.alloc` is fired to create the file handle and file buffer in the wasm context
- The main thread then creates and holds a `ReadableStream` for the `Blob`
- The main thread reads the file chunk by chunk, streaming it to the worker via `fs.write` messages
- Once streaming is finished, the `ReadableStream` is closed
- The model load is then triggered with `mmap = true`, and `mmap()` is wrapped to return a pointer to the correct data in the buffer allocated in step 1

The main downside of this approach is that on WebGPU, even though some tensors can be offloaded to the GPU, we still need to allocate the full model in main memory. For example, a 4GB model will still occupy 4GB of main memory, even if half of the layers (~2GB) are offloaded to the GPU.

## Function names in stack traces

The release wasm has no debug info, so `scripts/add_wasm_names.js` adds a standard wasm `name` section with cleaned names after each build. Browsers read this section themselves, so a stack trace shows `server_response::send` without any decoding in JS.

Emscripten's `--emit-symbol-map` flag produces a `.js.symbols` file mapping each wasm function index to its demangled C++ name. The script reads this file from the build directory and writes the result to `src/wasm/` or `compat/wasm/`. It always reads from the build directory, so running it twice does not add a second section.

The script runs as part of the docker build (see `scripts/docker-compose.yml`). It can also be run by hand:

```sh
node scripts/add_wasm_names.js build/wllama.wasm src/wasm/wllama.wasm
```

### Name cleaning rules

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

`dev/native-test/` contains a native harness that compiles `cpp/wllama-context.h` with llama.cpp's server sources and drives it like the JS worker does (single-threaded polling, glue-encoded messages). Use it to debug glue/server behavior with fast rebuilds and lldb, or to bisect whether a wasm failure comes from the C++ code or from the browser environment (WebGPU, JSPI, ...). See `dev/native-test/README.md`.

## Build process

The build process uses emscripten in docker to compile the project.

After compilation, `generate_glue_prototype.js` is called to generate the GLUE message types to be used in TypeScript.

Built wasm file will then be copied to the `src` directory.

Finally, `build_worker.sh` is called to generate the web worker code.
