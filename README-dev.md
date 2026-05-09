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

GLUE is a home-cook protocol inspired by Protobuf. It is used internally to communicate between wasm context and the javascript context of wllama.

The main goal of GLUE is to allow having a type-safe interface with low overhead. It works by using [Transferable objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects)

Upon build, `generate_glue_prototype.js` will be responsible of reading the `glue.hpp` file and produce the `glue/messages.ts` that can then be used from typescript.

### HeapFS

HeapFS is a ligh-weight wrapper around emscripten's default FS driver. The main goal is to allow `mmap()` operation to map to existing data, instead of copying it (the default behavior of emscripten). See `workers-code/llama-cpp.js` for more

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
