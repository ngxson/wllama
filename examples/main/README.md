# wllama main example

The chat app of llama.cpp ([`llama.cpp/ui`](../../llama.cpp/ui)), running on wllama: the models run in the browser, there is no server.

## How it works

The app talks to llama-server over HTTP. Here, `src/hooks.client.ts` replaces `window.fetch` before the app starts, and the requests to the llama-server API (`/props`, `/v1/models`, `/models/*`, `/v1/chat/completions`, ...) are answered in the page by wllama. The app runs in router mode: one model is loaded at a time, and a request for another model loads it.

The UI is not copied in this folder. `build.sh` copies `llama.cpp/ui` (same commit as the wasm) to `.build/`, applies `patches/`, copies `overlay/` on top, then builds it.

- `overlay/src/lib/wllama/`: the in-page server
  - `wllama-server.ts`: the model list, downloads, load and unload (`WllamaServer`)
  - `routes.ts`: the llama-server API routes, mapped onto `WllamaServer`
  - `fetch-shim.ts`: sends the API requests to `routes.ts`
  - `components/`: the Models page
- `overlay/src/routes/models/+page.svelte`: replaces the upstream models manager
- `overlay/vite.config.wllama.ts`: wraps the upstream `vite.config.ts`
- `patches/`: small changes to the upstream files, keep them as few as possible

## Build

Build wllama first (the app uses `esm/` of this repo), then:

```bash
npm run build   # output in dist/
npm run dev     # dev server; changes in overlay/ need a restart
```

Serve `dist/` with the `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers to use multithreading.

## Change the model list

The recommended models are in `models.json`, with the same params as `loadModelFromHF` (`repo`, and `quant` or `file`; `mmprojQuant` or `mmprojFile` to choose the mmproj, else Q8_0, F16 or BF16 is picked). Then run:

```bash
node update-models.mjs
```

It resolves the files like `loadModelFromHF`, reads the sizes and the capabilities (vision and audio from the mmproj, tools and thinking from the chat template, with the detectors of the upstream app), and writes `overlay/src/lib/wllama/models.generated.json`, sorted by size. The app only reads this file, so it does not call the HF API on each page load.

`quant` matches a part of the file name: when another file also contains it (`Q2_0` is in `PQ2_0`), use `file`.

## Add a patch

Change the file in `.build/ui`, then write the diff to `patches/` with paths relative to `ui/` (`a/apps/chat/...`, `b/apps/chat/...`). The patches are applied with `patch -p1` in file name order.
