# Release note Wllama V3.1

## What's new

Continuing from the [V3.0 release](./intro-v3.md), V3.1 continues to bring more interesting features into wllama. This release marks 2 major changes:
1. WebGPU support
2. Single WASM build (no more single/multi-threaded build)

### WebGPU support

WebGPU support is introduced via [PR #215](https://github.com/ngxson/wllama/pull/215). Currently only supports Chrome (for Firefox, a flag must be enabled manually).

Upon updating to V3.1, WebGPU will be enabled automatically. By default, all layers will be offloaded to GPU. If the model is too big to fit into VRAM, you can manually adjust the number of layers via the `n_gpu_layers` parameter of `LoadModelParams`. Example:

```js
await wllama.loadModel(files, {
  n_gpu_layers: 4, // meaning 4 layers are offloaded to GPU; set to 0 to disable GPU inference
});
```

### Single WASM build

From [PR #214](https://github.com/ngxson/wllama/pull/214), the separation between single-threaded build and multi-threaded build has been removed. Wllama now uses a single build that can support both single/multi-threaded and WebGPU, each feature can be toggled at runtime.

This allows cutting down the space to host the pre-built binary, while speeding up the build process.

To migrate from an older version:

```js
// Old config
const CONFIG_PATHS = {
  'single-thread/wllama.wasm': './path_to_source/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm' : './path_to_source/multi-thread/wllama.wasm',
};

// New config
const CONFIG_PATHS = {
  default: './path_to_source/wasm/wllama.wasm',
};
```
