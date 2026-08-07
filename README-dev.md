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

### Linear memory

The default build uses WebAssembly Memory64 with a 128 MiB initial memory and
a 16 GiB maximum. The compat build remains wasm32 and has a 4 GiB maximum.
Both builds allow memory growth. For multithreaded startup, the worker creates
the shared memory and can retry with a lower maximum on constrained devices.
Single-threaded startup uses Emscripten's imported-memory setup after Wllama
verifies that the browser can create a shared 16 GiB Memory64 descriptor.

Memory64 changes C/C++ pointers and `size_t` values to 64 bits. Values crossing
the JavaScript boundary therefore use `BigInt`, while offsets passed to
`TypedArray` and `Blob` APIs are converted to `Number`. This is exact throughout
the 16 GiB address range. Heap views must always be recreated after growth; use
`getHeapU8()` rather than caching an Emscripten `HEAPU8` view.

The browser maximum is a virtual-address and runtime ceiling, not an allocation
guarantee. A real model also needs memory for browser overhead, model input,
temporary buffers, and inference state.

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

## Compressed source map

Emscripten's `--emit-symbol-map` flag produces a `.js.symbols` file mapping each wasm function index to its demangled C++ name. `scripts/build_source_map.js` reads this file alongside the `.wasm` binary and produces a single TypeScript file (`src/wasm/source-map.ts`) containing a compact deduplicated name table per build, gzip-compressed and base64-encoded.

The script runs automatically as part of the docker build (see `scripts/docker-compose.yml`). It can also be run manually:

```sh
# uses build/ and build-compat/ by default
node scripts/build_source_map.js

# or with explicit paths
node scripts/build_source_map.js \
  --input default:build \
  --input compat:build-compat \
  --output src/wasm/source-map.ts
```

### Name cleaning rules

Raw demangled names can be hundreds of characters. The following rules are applied in order:

1. **std:: collapse** - any name starting with `std::` is replaced with the single hint `std::...`
2. **Lambda/closure extraction** - names containing `::$_N` or `::'lambda'` are replaced with the nearest enclosing context (the segment inside the last `<…>` before the marker)
3. **Parameter stripping** - parameter lists are dropped; empty `()` is kept, non-empty is removed entirely
4. **libc++ internals** - `::__1::`, `::__2::`, etc. are collapsed to `::`
5. **ABI tags** - `[abi:…]` annotations are removed
6. **Template truncation** - template argument content longer than 10 characters is truncated to `<first10chars...>`
7. **Final cleanup** - double `::::` collapsed, whitespace normalised

### Binary format (before gzip)

All integers are little-endian.

```
┌──────────────────────────────────────────────────────────┐
│ HEADER (12 bytes)                                        │
│   u32  first_func_id  - wasm function index of entry 0  │
│   u32  num_funcs      - number of functions              │
│   u32  num_names      - number of unique names           │
├──────────────────────────────────────────────────────────┤
│ NAME TABLE  (num_names entries)                          │
│   for each name:                                         │
│     u8   length       - byte length of name (max 254)   │
│     u8[] name         - UTF-8 string (no null term)      │
├──────────────────────────────────────────────────────────┤
│ INDEX ARRAY  (num_funcs × u16)                           │
│   u16  name_idx       - index into name table            │
│                         0xFFFF = no name / unknown       │
└──────────────────────────────────────────────────────────┘
```

To decode at runtime: base64-decode -> `DecompressionStream('gzip')` -> parse binary. Given a wasm function index `id`, look up `index_array[id - first_func_id]` to get the name table slot.
%
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

## Build process

The build process uses emscripten in docker to compile the project.

After compilation, `generate_glue_prototype.js` is called to generate the GLUE message types to be used in TypeScript.

Built wasm file will then be copied to the `src` directory.

Finally, `build_worker.sh` is called to generate the web worker code.

## Testing linear memory

After rebuilding both Wasm artifacts, run:

```sh
# Grow the real default artifact past 4 GiB, perform Wasm reads/writes on both
# sides of the boundary, then grow to and touch the final 16 GiB page.
npm run test:memory64

# Force the locally built wasm32 compat worker and run real model inference.
npm run test:compat

# Run the ordinary default-build browser suite.
npm run test:auto
```

The Memory64 boundary test is sparse and suitable for a normal 64-bit Chromium
runner. The real-model stress runner uses an explicit 12 GiB test budget. Serve
tests with COOP and COEP headers so the shared-memory path is exercised.

### Real-model Chromium stress test

The Memory64 stress lab exercises the package as a user would: it downloads a
real GGUF into the browser model cache, loads it through the default Memory64
artifact, validates model metadata, and generates deterministic tokens. The
Playwright runner connects through Chrome DevTools Protocol and records page and
worker console output, exceptions, crashes, failed requests, screenshots, a
trace, and process-tree RSS/PSS samples.

```sh
# Fast end-to-end validation with the 18.2 MiB TinyLlama fixture.
npm run test:memory64:stress:smoke

# Sequential 4.36 GiB, 8.37 GiB, and 11.29 GiB model runs.
npm run test:memory64:stress

# Use separate download and inference browser lifetimes so cached model pages
# can be reclaimed before tensors are committed.
npm run test:memory64:stress:low-memory

# Exercise the shared-Memory64 pthread path instead of the default one-thread
# physical stress configuration.
npm run test:memory64:stress:multithread
```

The real-model stress runner requires Linux and 64-bit Chromium 137 or newer.
It reads Linux `/proc` process data to prove that large fixtures were physically
resident instead of merely reserved in virtual memory. Every current fixture
fits within the runner's 12 GiB test budget. Host and cgroup memory values are
recorded for diagnostics but do not gate a run: cgroup usage includes
reclaimable page cache and can substantially understate usable memory after a
model download. Selecting a fixture larger than the declared budget is a
configuration error. Each model runs in a fresh browser so OPFS data and Wasm
memory from the prior tier cannot influence the next result.

The low-memory command stores its temporary Chromium profiles on the workspace
volume rather than Docker's overlay filesystem. It first downloads each model,
closes Chrome, flushes the OPFS files, and then opens a fresh renderer for model
loading and inference. The runner raises the isolated test origin's quota to
32 GiB through Chrome DevTools Protocol and removes the profile after the
inference phase. The inference renderer must rediscover valid cached shards
without remote model requests. A pass requires exact fixture bytes, valid model
metadata, expected fixture output, the requested threading mode, zero fatal
DevTools events, and measured browser PSS at least as large as the selected
fixture for every model over 4 GiB. This avoids overlapping reclaimable download
cache with the physical tensor allocation; it does not reduce the model's Wasm
memory use.

To use the proof-of-concept manually, build the browser package, run
`npm run serve:mt`, and open
`http://localhost:8080/examples/memory64/`. The page includes the same model
presets and mirrors package logs while the browser developer console remains
the source of truth.
