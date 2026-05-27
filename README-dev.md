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
- 4 bytes — magic number (`GLUE`)
- 4 bytes — version number (`GLUE_VERSION`)
- 8 bytes — message prototype ID
- 4 bytes — message length (unsigned)
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

The standard `.wasm.map` produced by emscripten maps every wasm byte offset to a source file and line. For wllama's purposes — resolving a stack-trace function index like `wasm-function[2436]` to a file and line — this is far more data than needed, and it includes noise from emsdk/libc/libc++ that is never actionable.

`scripts/build_source_map.js` post-processes one or more `.wasm.map` files into a single TypeScript file (`src/wasm/source_map.ts`) containing a compact binary source map per build, gzip-compressed and base64-encoded. The corresponding `.wasm` file must be next to each `.map` file (same base name, minus `.map`).

```sh
node scripts/build_source_map.js \
  --input default:build/wllama.wasm.map \
  --input compat:build-compat/wllama.wasm.map \
  --output src/wasm/source_map.ts
```

### What it does

- Parses the wasm binary to find the byte offset of every function body
- Looks up each offset in the source map to get `(file, line)`
- Drops all emsdk/libc/libc++ functions (marked as no-mapping)
- Shortens source file paths to minimal disambiguating suffixes (e.g. `models/qwen3vl.cpp` instead of the full path)
- Encodes the result as a compact binary, then gzip-compresses and base64-encodes it

### Size

| | Standard `.wasm.map` | Compressed output |
|---|---|---|
| Raw size | ~5 MB | ~22 KB binary |
| Shipped size | — | ~9 KB (gzipped) / ~12 KB (base64 in TS) |
| Granularity | per instruction | per function |
| emsdk entries | included | dropped |

### Binary format (before gzip)

All integers are little-endian.

```
┌─────────────────────────────────────────────────────────┐
│ HEADER (12 bytes)                                       │
│   u32  first_func_id   — wasm function index of entry 0 │
│   u32  num_funcs       — number of functions            │
│   u32  num_sources     — number of source files         │
├─────────────────────────────────────────────────────────┤
│ SOURCE NAME TABLE                                       │
│   for each source (num_sources entries):                │
│     u8   length        — byte length of name            │
│     u8[] name          — UTF-8 filename (no null term)  │
├─────────────────────────────────────────────────────────┤
│ FILE INDEX STREAM  (RLE)                                │
│   repeated until num_funcs values decoded:              │
│     u16  run_length    — how many consecutive functions │
│     u16  file_idx      — index into source name table   │
│                          0xFFFF = no mapping            │
├─────────────────────────────────────────────────────────┤
│ LINE NUMBER STREAM                                      │
│   u16[num_funcs]       — source line per function       │
│                          (0 when file_idx == 0xFFFF)    │
└─────────────────────────────────────────────────────────┘
```

To decode at runtime: base64-decode → `DecompressionStream('gzip')` → parse binary. Given a function index `id`, the entry is at position `id - first_func_id` in both streams.

## Build process

The build process uses emscripten in docker to compile the project.

After compilation, `generate_glue_prototype.js` is called to generate the GLUE message types to be used in TypeScript.

Built wasm file will then be copied to the `src` directory.

Finally, `build_worker.sh` is called to generate the web worker code.
