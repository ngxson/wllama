# @wllama/wllama-compat

Optional package that provides the compat WASM build of `@wllama/wllama`, for browsers that lack [JSPI](https://github.com/WebAssembly/js-promise-integration) or [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd) support - most notably Safari.

## Why this package exists

The default `@wllama/wllama` build relies on two modern WebAssembly features: [JSPI](https://github.com/WebAssembly/js-promise-integration) and [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd).

When one of them is missing, wllama automatically falls back to **compat mode**: a separate WASM build that uses [Asyncify](https://emscripten.org/docs/porting/asyncify.html) instead of JSPI, and drops relaxed SIMD. It supports the same features as the default build, including WebGPU and models bigger than 4GB, but it is slower.

## Browser compatibility

| | Chromium | Firefox | Safari |
|---|---|---|---|
| Auto-compat (default, recommended) | ✅ | ✅ (supports WebGPU) | 🟡 (supports WebGPU) |
| Non-compat mode | ✅ | ✅ (supports WebGPU) | ❌ |

- ✅: Good speed
- 🟡: Acceptable speed
- ❌: Does not run at all

Chromium and Firefox run the default build. An older Firefox without JSPI still runs the default build, but without WebGPU and models over 4GB; it uses compat mode only if it does not support relaxed SIMD.

## Default behaviour

Out of the box, wllama fetches the compat build from jsDelivr CDN when compat mode is needed, nothing needs to be done. If you want to host the file yourself (no external CDN dependency), install this package, see [Using this package](#using-this-package).

To opt out of compat mode completely (e.g. you don't target Safari):

```ts
wllama.setCompat(null);
```

## Using this package

**You only need to install this package if you want to host the compat build yourself**. By default, it is pulled from CDN.

```bash
npm install @wllama/wllama-compat
```

Then copy `node_modules/@wllama/wllama-compat/wasm/wllama.wasm` to your public directory and call `setCompat()` with the URL pointing to that file:

```ts
import { Wllama } from '@wllama/wllama';

const wllama = new Wllama({ default: '/wasm/wllama.wasm' });

wllama.setCompat({
  wasm: '/wllama-compat/wasm/wllama.wasm',
});
```

For Vite:

```ts
import { Wllama } from '@wllama/wllama';
import wasm from '@wllama/wllama/esm/wasm/wllama.wasm?url';
import compatWasm from '@wllama/wllama-compat/wasm/wllama.wasm?url';

const wllama = new Wllama({ default: wasm });
wllama.setCompat({ wasm: compatWasm });
```

> [!IMPORTANT]
> The version of `@wllama/wllama-compat` must be the same as `@wllama/wllama`. Otherwise, loading a model fails with a `version_mismatch` error.

## Migration from older versions

- The JS code of the build is now embedded inside `wllama.wasm`, so there is only one file to host. `setCompat()` does not need a `worker` (the `wllama.js` file) anymore. `wllama.js` is kept as a placeholder, so old setups still work (with a warning). A `worker` with real emscripten code still overrides the embedded JS code, for example for custom builds.
- The second parameter of `setCompat()` (`'safari'` or `'firefox_safari'`) is not used anymore. Firefox now runs the default build, see [Browser compatibility](#browser-compatibility).
