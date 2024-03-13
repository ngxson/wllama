# wllama - WASM binding for llama.cpp

![](./README_banner.png)

Another WASM binding for [llama.cpp](https://github.com/ggerganov/llama.cpp). Inspired by [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm), but unlike it, **wllama** aims to supports **low-level API** like (de)tokenization, embeddings,...

WASM allow llama.cpp to run directly on browser, without any server part.

## Demo

- Basic usages with completions and embeddings: https://ngxson.github.io/wllama/examples/basic/

## How to build

```shell
# Require having docker compose installed
# Firstly, build llama.cpp into wasm
./scripts/build_wasm.sh
# (Optionally) Build ES6 module
npm run build
```

## How to use

See in `examples`

```javascript
import { Wllama } from '../../esm/index.js';

(async () => {
  // Automatically switch between single-thread and multi-thread version based on browser support
  // If you want to enforce single-thread, remove "wasmMultiThreadPath" and "workerMultiThreadPath"
  const wllama = new Wllama({
    wasmSingleThreadPath: '../../esm/single-thread/wllama.wasm',
    wasmMultiThreadPath: '../../esm/multi-thread/wllama.wasm',
    workerMultiThreadPath: '../../esm/multi-thread/wllama.worker.mjs',
  });
  await wllama.loadModel('https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf', {});
  const outputText = await wllama.createCompletion(elemInput.value, {
    nPredict: 50,
    sampling: {
      temp: 0.5,
      top_k: 40,
      top_p: 0.9,
    },
  });
  console.log(outputText);
})();
```

## TODO

- Better documentation
- Support multi-sequences: knowing the resource limitation when using WASM, I don't think having multi-sequences is a good idea
- Multi-modal: Waiting for refactoring LLaVA implementation from llama.cpp
