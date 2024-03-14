# wllama - Wasm binding for llama.cpp

![](./README_banner.png)

Another WebAssembly binding for [llama.cpp](https://github.com/ggerganov/llama.cpp). Inspired by [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm), but unlike it, **wllama** aims to supports **low-level API** like (de)tokenization, embeddings,...

Wasm allow llama.cpp to run directly on browser, without any server part.

## Features

- Typescript support
- No runtime dependency (see [package.json](./package.json))
- High-level API: completions, embeddings
- Low-level API: (de)tokenize, KV cache control, sampling control,...
- Ability to load splitted model
- Auto switch between single-thread and multi-thread build based on browser support
- Pre-built npm package [@wllama/wllama](https://www.npmjs.com/package/@wllama/wllama)

## Demo and documentations

**Documentation:** https://ngxson.github.io/wllama/docs/

Demo:
- Basic usages with completions and embeddings: https://ngxson.github.io/wllama/examples/basic/

## How to use

### Use Wllama inside React Typescript project

Install it:

```bash
npm i @wllama/wllama
```

For complete code, see [examples/reactjs/src/App.tsx](./examples/reactjs/src/App.tsx)

NOTE: this example only covers completions usage. For embeddings, please see [examples/basic/index.html](./examples/basic/index.html)

### Simple usage with ES6 module

For complete code, see [examples/basic/index.html](./examples/basic/index.html)

```javascript
import { Wllama } from './esm/index.js';

(async () => {
  const CONFIG_PATHS = {
    'single-thread/wllama.wasm'     : './esm/single-thread/wllama.wasm',
    'multi-thread/wllama.wasm'      : './esm/multi-thread/wllama.wasm',
    'multi-thread/wllama.worker.mjs': './esm/multi-thread/wllama.worker.mjs',
  };
  // Automatically switch between single-thread and multi-thread version based on browser support
  // If you want to enforce single-thread, add { "n_threads": 1 } to LoadModelConfig
  const wllama = new Wllama(CONFIG_PATHS);
  await wllama.loadModelFromUrl('https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf', {});
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

## How to build

This repository already come with pre-built binary. But if you want to build it yourself, you can use the commands below:

```shell
# Require having docker compose installed
# Firstly, build llama.cpp into wasm
npm run build:wasm
# (Optionally) Build ES6 module
npm run build
```

## TODO

- Guide: How to split gguf file?
- Support multi-sequences: knowing the resource limitation when using WASM, I don't think having multi-sequences is a good idea
- Multi-modal: Waiting for refactoring LLaVA implementation from llama.cpp
