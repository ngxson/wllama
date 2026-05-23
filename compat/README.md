# @wllama/wllama-compat

Optional package that provides compatibility WASM assets for `@wllama/wllama` on browsers that lack [JSPI](https://github.com/WebAssembly/js-promise-integration) or [MEMORY64](https://github.com/WebAssembly/memory64) support - most notably Safari and older browsers.

## Why this package exists

The default `@wllama/wllama` build relies on two modern WebAssembly features: [JSPI](https://github.com/WebAssembly/js-promise-integration) and [MEMORY64](https://github.com/WebAssembly/memory64)

When either feature is absent, wllama automatically falls back to **compat mode**: a separate WASM build that uses [Asyncify](https://emscripten.org/docs/porting/asyncify.html) instead of JSPI, and drops MEMORY64.

> **Note:** Compat mode has significantly lower performance than the default build. Use it only as a fallback.

## Browser compatibility

| | Chromium | Firefox | Safari |
|---|---|---|---|
| Auto-compat (default, recommended) | ✅ | 🟡 (no WebGPU) | 🟡 (supports WebGPU) |
| Force-compat | ✅ | 🔴 (supports WebGPU) | 🟡 (supports WebGPU) |
| Non-compat mode | ✅ | 🟡 (no WebGPU) | ❌ |

- ✅: Good speed
- 🟡: Acceptable speed
- 🔴: Runs but slow, not usable
- ❌: Does not run at all

### Default behaviour

Out of the box, wllama fetches the compat assets from jsDelivr CDN when compat mode is needed. If you want to self-host the assets (no external CDN dependency), install this package (see new section.)

### Recommended preset

By default (`mode = 'safari'`), compat is disabled on Firefox because WebGPU via compat mode is extremely slow there. This is the recommended behaviour:

```js
wllama.setCompat('default');
```

If you also want compat on Firefox (e.g. to reach users without JSPI enabled), pass `'firefox_safari'`:

```js
wllama.setCompat('default', 'firefox_safari');
```

## Disabling compat mode

To opt out of compat mode completely (e.g. you don't target Safari):

```ts
wllama.setCompat(null);
```

## Using this package

**You only need to install package if you want to store compat assets locally**. By default, assets are pulled from CDN.

```bash
npm install @wllama/wllama-compat
```

Then copy the assets from `node_modules/@wllama/wllama-compat/wasm/` to your public directory and call `setCompat()` with the URLs pointing to those files:

```ts
import { Wllama } from '@wllama/wllama';

const wllama = new Wllama({ default: '/wasm/wllama.wasm' });

wllama.setCompat({
  wasm: '/wllama-compat/wasm/wllama.wasm',
  worker: '/wllama-compat/wasm/wllama.js',
});
```

**IMPORTANT**: for Vite, you will need to import the JS as `?raw`

```ts
import compatWasm from '@wllama/wllama-compat/wasm/wllama.wasm?url';
import compatWorker from '@wllama/wllama-compat/wasm/wllama.js?raw'; // IMPORTANT: ?raw, NOT ?url

export const WLLAMA_COMPAT_CONFIG = {
  wasm: compatWasm,
  worker: {
    code: compatWorker,
  },
};

const instance = new Wllama(WLLAMA_CONFIG_PATHS, { logger: DebugLogger });
instance.setCompat(WLLAMA_COMPAT_CONFIG);
```
