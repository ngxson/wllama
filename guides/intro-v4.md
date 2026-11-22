# Introducing Wllama V4.0

## What's new

V4.0 is a big step for performance and model size. Inference is more than 2x faster, and the size limit of a model is gone: a GGUF file of any size can be loaded as-is. Key highlights:

- Up to 2.8x faster prompt processing and 2.5x faster generation (Chrome, CPU)
- No more model size limit: models bigger than 4GB are spread across multiple wasm instances
- Mixed CPU + WebGPU inference for models that do not fit in the GPU
- Faster model loading from the OPFS cache, and faster downloads over multiple connections
- Speculative decoding: MTP, DFlash, DSpark and Eagle3
- LoRA adapters
- Full-duplex realtime audio, video input, `saveSlotState` / `restoreSlotState`, and `tokenize` / `detokenize`

The full benchmark (Chrome, Firefox and Safari) can be found in [this comment](https://github.com/ngxson/wllama/issues/338#issuecomment-7062438290).

## Faster inference

The wasm build now uses the SIMD and relaxed SIMD kernels of llama.cpp ([PR #331](https://github.com/ngxson/wllama/pull/331)), and the main build uses 32-bit memory instead of Memory64 ([PR #347](https://github.com/ngxson/wllama/pull/347)). Compared to v3.13.0, on Chrome (CPU, 6 threads, Apple M5 Max):

| Model | Prompt processing | Generation |
|---|---|---|
| LFM2 350M Q4_K_M | 247 → 598 t/s (2.4x) | 135 → 277 t/s (2.1x) |
| LFM2.5 1.2B Q4_K_M | 73 → 207 t/s (2.8x) | 49 → 118 t/s (2.4x) |
| Qwen2.5-Omni 3B Q4_K_M | 25 → 66 t/s (2.6x) | 17 → 42 t/s (2.5x) |

Firefox gets a similar gain (2.0x to 2.3x). Safari uses the compat build, the gain is smaller (up to 1.4x).

No code change is needed, just update the package.

## No more model size limit

A wasm instance cannot use more than 4GB of memory. From V4.0, wllama handles this automatically:

- A GGUF file of 4GB or more is split into virtual shards in memory, no data is copied ([PR #347](https://github.com/ngxson/wllama/pull/347))
- If the part of the model on the CPU does not fit in one wasm instance, the layers are spread across extra wasm instances (RPC workers), each one has its own 4GB memory ([PR #352](https://github.com/ngxson/wllama/pull/352))
- With WebGPU, the layers that do not fit in the GPU go to the RPC workers, the placement is done by the "fit" logic of llama.cpp ([PR #358](https://github.com/ngxson/wllama/pull/358))

This means you can now load the GGUF file as-is, there is no need to split it anymore:

```js
// a 5GB model, a single file
await wllama.loadModelFromUrl('https://huggingface.co/.../gemma-4-E4B-it-Q4_0.gguf');
await wllama.exit();

// a part on WebGPU, the rest on the CPU
await wllama.loadModelFromUrl('https://huggingface.co/.../gemma-4-E4B-it-Q4_0.gguf', {
  n_gpu_layers: 20,
});
```

You can control the number of RPC workers with `rpc_workers`. By default, they are only used when needed; set it to 0 to disable them.

> [!IMPORTANT]
> RPC workers need multi-threading, so the page must be cross-origin isolated (COOP / COEP headers). Without it, a model whose CPU part is bigger than 4GB cannot be loaded. An older Firefox without JSPI cannot load it either.

## Faster loading and downloading

Models in the OPFS cache are now read with a sync access handle, directly into the wasm memory ([PR #356](https://github.com/ngxson/wllama/pull/356)). On Chrome, loading a 2GB model is about 2.8x faster than v3.13.0.

Downloading a single big file was limited to one connection, even with `parallelDownloads`. From V4.0, a file of 64MB or more is downloaded in parts over multiple connections ([PR #364](https://github.com/ngxson/wllama/pull/364)), and an interrupted download continues where it stopped ([PR #336](https://github.com/ngxson/wllama/pull/336)). The number of connections per file can be changed via `connections`:

```js
await wllama.loadModelFromUrl('https://example.com/model.gguf', {
  connections: 8, // default: 4
});
```

## Speculative decoding

Speculative decoding uses a small draft model to guess the next tokens, then the main model checks them all at once ([PR #362](https://github.com/ngxson/wllama/pull/362)). Supported drafts are: MTP, DFlash, DSpark, Eagle3 and a smaller model of the same family (draft-simple). The type is detected from the draft file:

```js
// a draft file: DFlash, DSpark, Eagle3, an MTP sidecar, or a smaller model
await wllama.loadModelFromUrl({
  url: 'https://huggingface.co/.../Qwen3-4B-Q4_K_M.gguf',
  draftUrl: 'https://huggingface.co/.../dflash-Qwen3-4B-Q4_K_M.gguf',
});
await wllama.exit();

// a model with its MTP layers inside: no draft file
await wllama.loadModelFromUrl('https://huggingface.co/.../Qwen3.5-4B-MTP-Q4_K_M.gguf', {
  spec_type: 'draft-mtp',
});
```

With `loadModel()`, a DFlash, DSpark or Eagle3 draft, or an MTP sidecar, can also be passed in the list of files, along with the model; a smaller model used as the draft goes to `spec_draft_model`. On WebGPU, DFlash, DSpark and MTP make generation about 1.2x faster. On CPU, it is usually slower, so it is not recommended.

## LoRA adapters

LoRA adapters can be loaded with the model ([PR #362](https://github.com/ngxson/wllama/pull/362), [PR #363](https://github.com/ngxson/wllama/pull/363)):

```js
// from URLs, applied with scale 1
await wllama.loadModelFromUrl({
  url: 'https://example.com/model.gguf',
  loraUrls: ['https://example.com/adapter.gguf'],
});
await wllama.exit();

// from a Blob, with its scale
await wllama.loadModelFromUrl('https://example.com/model.gguf', {
  lora_adapters: [{ blob: adapterBlob, scale: 0.5 }],
});

// change the scale for a request (id is the index of the adapter)
await wllama.createChatCompletion({
  messages: [{ role: 'user', content: 'Hello!' }],
  lora: [{ id: 0, scale: 0 }],
});
```

## Other additions

- Full-duplex realtime audio, for models like Hibiki (French to English speech translation): the model speaks while it listens, there are no turns. Check `has_audio_duplex` in `getLoadedContextInfo()` ([PR #361](https://github.com/ngxson/wllama/pull/361))
- Video input: see the [video input section](./intro-v3.md#video-input) of the V3 guide ([PR #324](https://github.com/ngxson/wllama/pull/324))
- `saveSlotState()` / `restoreSlotState()` to save the processed prompt and reuse it later ([PR #326](https://github.com/ngxson/wllama/pull/326))
- `tokenize()` and `detokenize()` are back, with the same options as llama-server ([PR #360](https://github.com/ngxson/wllama/pull/360))
- New TTS options: `temperature`, `speaker_ref_text`, `raw_prompt`, `speed`, etc.
- Decision models (`createSystemOne`): LFM3-D1 packs up to 16 questions in one prompt, CLEF answers all of them in one prompt

## Migration from v3.x

There is no breaking change in the API. However, some behaviors have changed.

### Splitting the model is not needed anymore

Previously, we recommended splitting the model into chunks of 512MB. From V4.0, we recommend loading the GGUF file as-is. Models that are already split with `llama-gguf-split` are still supported.

### Compat build

Firefox runs the main build, its speed is now close to Chrome. It only falls back to the compat build if it does not support wasm relaxed SIMD (older versions). Because of that, the `firefox_safari` mode of `setCompat` is not needed anymore, the second parameter is ignored.

The JS code of the worker is also embedded inside the wasm file ([PR #327](https://github.com/ngxson/wllama/pull/327)), so only the wasm file is needed:

Previously in v3.x:

```js
wllama.setCompat({
  wasm: './path_to_source/compat/wasm/wllama.wasm',
  worker: './path_to_source/compat/wasm/wllama.js',
}, 'firefox_safari');
```

From V4.0:

```js
wllama.setCompat({
  wasm: './path_to_source/compat/wasm/wllama.wasm',
});
```

### Changes in `ModelManager` and `CacheManager`

- `ModelSource` gains `draftUrl` and `loraUrls`
- `getModelOrDownload()` now checks the extra files too: a cached model is only reused if its `mmprojUrl`, `draftUrl` and `loraUrls` are the same
- When the storage supports it (OPFS), the first request of a download has a `Range: bytes=0-` header, to know if the server supports ranges

### Internal changes

- The main build uses 32-bit memory; Memory64 is not used anymore
- The test suite also runs the compat build on CI
- A dev server to test on a phone or tablet: see "Testing on a phone or tablet" in [README-dev.md](../README-dev.md)
