# wllama

Wllama is a webassembly binding of llama.cpp. It contains the main source code of llama.cpp compiled to wasm (with emscripten), plus a wrapper to provide various convenient APIs, including: downloading and caching models, compatibility, etc.

## Project structure

The project has these dir:
- `src`: the main typescript source code
- `cpp`: C++ interface
- `scripts`: various scripts for development
- `examples`: various examples

The project has these main components:
- `wllama.ts`: the main public API
- `model-manager.ts`: relies on cache manager to manage models. For ex, a model can compose from multiple files
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

### HeapFS

HeapFS is a ligh-weight wrapper around emscripten's default FS driver. The main goal is to allow `mmap()` operation to map to existing data, instead of copying it (the default behavior of emscripten). See `workers-code/llama-cpp.js` for more details.

## Startup process

Upon started, these steps are performed:
- `ProxyToWorker` is created in the main wllama JS context
- A web worker is spawned, the code is taken from `workers-code/generated.ts`
- The worker loads emscripten code, setup the environement then eventually call the `main()` inside `wllama.cpp`. These preparation steps are injected (see `llama-cpp.js`):
    - Hooking `printf` functions
    - Setup HeapFS
    - Setup communication callbacks

## Build process

The build process uses emscripten in docker to compile the project.

After compilation, `generate_glue_prototype.js` is called to generate the GLUE message types to be used in typescript.

Built wasm file will then be copied to the `src` directory.

Finally, `build_worker.sh` is called to generate the web worker code.
