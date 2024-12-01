# Introducing Wllama V2.0

## What's new

V2.0 introduces significant improvements in model management and caching. Key features include:

- Completely rewritten model downloader with service worker
- New `ModelManager` class providing comprehensive model handling and caching capabilities
- Enhanced testing system built on the `vitest` framework

## Added `ModelManager`

The new `ModelManager` class provides a robust interface for handling model files:

```typescript
// Example usage
const modelManager = new ModelManager();

// List all models in cache
const cachedModels = await modelManager.getModels();

// Add a new model
const model = await modelManager.downloadModel('https://example.com/model.gguf');

// Check if model is valid (i.e. it is not corrupted)
// If status === ModelValidationStatus.VALID, you can use the model
// Otherwise, call model.refresh() to re-download it
const status = await model.validate();

// Re-download if needed (useful when remote model file has changed)
await model.refresh();

// Remove model from cache
await model.remove();

// Get model files as Blobs
// Then, you can load the blobs into Wllama instance
const blobs = await model.open();
const wllama = new Wllama(CONFIG_PATHS);
await wllama.loadModel(blobs);

// Alternatively, you can also pass directly model URL like in v1.x
// This will automatically download the model to cache
await wllama.loadModelFromUrl('https://example.com/model.gguf');
```

Key features of `ModelManager`:
- Automatic handling of split GGUF models
- Built-in model validation
- Parallel downloads of model shards
- Cache management with refresh and removal options

## Migration to v2.0

### Simplified `new Wllama()` constructor

In v2.0, the configuration paths have been simplified. You now only need to specify the `*.wasm` files, as the `*.js` files are no longer required.

Previously in v1.x:

```js
const CONFIG_PATHS = {
  'single-thread/wllama.js'       : '../../esm/single-thread/wllama.js',
  'single-thread/wllama.wasm'     : '../../esm/single-thread/wllama.wasm',
  'multi-thread/wllama.js'        : '../../esm/multi-thread/wllama.js',
  'multi-thread/wllama.wasm'      : '../../esm/multi-thread/wllama.wasm',
  'multi-thread/wllama.worker.mjs': '../../esm/multi-thread/wllama.worker.mjs',
};
const wllama = new Wllama(CONFIG_PATHS);
```

From v2.0:

```js
// You only need to specify 2 files
const CONFIG_PATHS = {
  'single-thread/wllama.wasm': '../../esm/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm' : '../../esm/multi-thread/wllama.wasm',
};
const wllama = new Wllama(CONFIG_PATHS);
```

Alternatively, you can use the `*.wasm` files from CDN:

```js
import WasmFromCDN from '@wllama/wllama/esm/wasm-from-cdn.js';
const wllama = new Wllama(WasmFromCDN);
// NOTE: this is not recommended, only use when you can't embed wasm files in your project
```

In addition to this, the constructor of `Wllama` also accepts second param of type `WllamaConfig`.

NOTE: Most of these parameters are moved from `DownloadModelConfig` which is used by `wllama.loadModelFromUrl`

Example:

```js
const wllama = new Wllama(CONFIG_PATHS, {
  parallelDownloads: 5, // maximum files to be downloaded at the same time
  allowOffline: false,
});
```

### `Wllama.loadModelFromUrl`

As mentioned earlier, some options are moved to `Wllama` constructor, including:
- `parallelDownloads`
- `allowOffline`

### Other changes

- `Wllama.downloadModel` is removed. Please use `ModelManager.downloadModel` instead
- `loadModelFromUrl` won't check if cached model is up-to-date. You may need to manually call `Model.refresh()` to re-download the model.
- Changes in `CacheManager`:
  - Added `CacheManager.download` function
  - `CacheManager.open(nameOrURL)` now accepts both file name and original URL. It now returns a `Blob` instead of a `ReadableStream`