# native-test

A native (non-wasm) harness that compiles `cpp/wllama-context.h` together with llama.cpp's server sources and drives it exactly like the JS worker does: single-threaded, one `run_loop()` iteration per `get_result` poll, glue-encoded messages, per-request response readers.

## Why

Debugging the C++ glue through the emscripten/browser stack is slow and mostly blind. This harness runs the same code natively, so you get fast rebuilds, `printf`/`lldb`, and clean stderr ordering.

It is also a **bisection tool** for wasm-only failures:

- Bug reproduces here -> it is in `cpp/wllama-context.h` or in llama.cpp's server code. Debug it natively.
- Bug only happens in the browser -> suspect the wasm environment instead: the WebGPU backend, JSPI/asyncify, emscripten build flags, worker interleaving. (Example: the ggml-webgpu multi-output bug from issue #261 passed here but crashed in wasm.)

## Build

```sh
cmake -S dev/native-test -B dev/native-test/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build dev/native-test/build --target native-test -j
```

## Run

Any small GGUF works; the test suite's tiny model is a good default:

```sh
curl -sL -o /tmp/stories15M-q4_0.gguf \
  "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf"
dev/native-test/build/native-test /tmp/stories15M-q4_0.gguf
```

Expected output ends with `== ALL OK`.

## Adapting it

`main.cpp` is meant to be edited per investigation:

- The default scenario posts request A, polls it 4 times (mid-generation), then posts B and C - forcing mixed generation+prompt batches and multi-slot decoding. Change the interleaving to reproduce other patterns.
- `call_action()` + `set_all_null()` mimic the JS glue encoder: null every field first, then set `value` **and** `dtype` for each field you need (fields default to "present", and e.g. an empty `pooling_type` string would be parsed and rejected).
- The stubs at the top replace emscripten-only symbols and the build-info / `common_log` functions that the wasm link resolves differently; if the linker reports a new undefined symbol after a llama.cpp sync, add a stub for it there.
- Keep the source lists in `CMakeLists.txt` in sync with the root `CMakeLists.txt` when the wasm build's file lists change.
