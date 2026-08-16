# SharedWorker mode (issue #256)

Status: implemented and tested (phases 1-5 done). This doc records the design, the browser facts it is built on, and where everything lives, so another agent (or future me) can take over.

## Goal

One model instance shared by all tabs of the same origin. First tab loads the model, later tabs attach to it. Requested in https://github.com/ngxson/wllama/issues/256

Usage: `new Wllama(paths, { sharedWorker: true })`, everything else is unchanged public API.

## Browser facts (verified empirically in Chrome stable, 2026-08)

These facts drove the whole design. Re-verify them if browsers changed.

| Capability inside SharedWorkerGlobalScope | Chrome | Firefox | WebKit |
|---|---|---|---|
| `new Worker()` | NO ("Worker is not defined") | yes | NO (until Safari 27) |
| `new SharedWorker()` | NO | NO | NO |
| `SharedArrayBuffer` / `crossOriginIsolated` | NO, even with COOP/COEP | NO | NO |
| WebGPU (`navigator.gpu.requestAdapter()`) | YES, real adapter | exposed, null adapter | untested |
| JSPI (`WebAssembly.Suspending`) | YES | NO (same as page) | untested |
| OPFS `createWritable()` (async) | YES | YES | untested |
| OPFS `createSyncAccessHandle()` | NO (dedicated worker only) | NO | NO |
| `importScripts()` | YES (classic worker) | YES | - |

Other verified facts:
- A SharedWorker is deduplicated by (origin, name, URL string). Attaching to a RUNNING worker via a blob URL string works even after the tab that created the blob died (no re-fetch happens on attach). Starting a NEW worker from a stale blob URL fails with a script load error, which is detectable via the error event.
- SharedWorker lifetime = document connections only. Holding a MessagePort to it does not keep it alive.
- There is no event when a client tab closes. Liveness needs heartbeats.
- The emscripten glue (src/wasm/wllama.js) contains no import.meta / export, so it is classic-script-safe and runs under importScripts.

## Design

```
tab A: Wllama -- ProxyToSharedWorker --+
                                       +-- SharedWorker scope: rpc server -- ProxyToWorker -- in-scope module (importScripts) -- wasm
tab B: Wllama (hydrated)  -------------+       + state store (snapshot + modelId) + init state machine + heartbeats
```

Key decisions and why:

1. The wasm module runs IN the SharedWorker scope itself (no nested worker, impossible in Chrome anyway). The dedicated worker in classic mode exists only to get off the main thread; a SharedWorker is already off the main thread.
2. The real ProxyToWorker runs in the scope, tabs use a thin rpc stub (ProxyToSharedWorker) with the same 5-method surface (moduleInit, wllamaStart, wllamaAction, wllamaDebug, wllamaExit). This keeps task-id assignment and the JSPI action queue in one place, so no id namespacing is needed.
3. Single-thread CPU is forced (nbThread=0), because no SAB. WebGPU works normally, this mode is WebGPU-first.
4. The Wllama class stays on the tab. After load, the derived model state is pushed to the scope as a snapshot `{ loadResult: GlueMsgLoadRes, params: {seed, embeddings, pooling_type, default_template_kwargs} }`. A joining tab replays `setLoadedState(loadResult, params)` from the snapshot, skipping module init and load entirely. This means zero per-field serialization code to maintain: new derived props automatically ride along via loadResult.
5. No cross-tab completion lock. The engine is a slot-based server (n_parallel, req_id-tracked get_result polling), concurrent completions from different tabs are natively safe. Single glue actions are serialized by the scope task queue.
6. Everything is blob-based, no extra hosted file. Discovery of the SharedWorker between tabs:
   - localStorage[`wllama-sw-url::<version-tag>`] holds the blob URL of the running scope
   - attach by URL string works while the scope is alive; if attach fails, mint a fresh blob URL under `navigator.locks` (prevents two cold tabs creating two scopes) and overwrite the registry
   - version tag = LIBLLAMA_VERSION + hash(scope code), so tabs on different deployed builds never share a scope

Rejected alternatives (investigated earlier, do not redo):
- Nested worker via "ask a tab to spawn it and transfer a MessagePort": works (verified), but pointless once the module runs in-scope. Two-SharedWorker topology (broker + wasm host) also works but adds a keep-alive/rewire protocol for no capability gain.
- Cross-tab op lock via Web Locks: dropped, see decision 5.

## Runtime protocol

Tab <-> scope messages: `{ id, verb, payload }` request, `{ id, result }` or `{ id, err }` response. Broadcast events without id: `{ evt: 'hello' | 'log' | 'ready' | 'dying' }`.

Verbs: `hb`, `status`, `module-init`, `start`, `action`, `debug`, `get-state`, `set-state`, `exit` (kill scope for all tabs), `reset` (kill without wllama_exit).

Scope state machine: `uninit -> initializing(port) -> ready`. Rules:
- only one tab can run module-init (others get "cannot init..." and must waitReady + hydrate)
- set-state flips initializing -> ready and broadcasts 'ready'
- tabs heartbeat every 1s; scope prunes ports silent for 6s; if the pruned port was the initializer while initializing, the scope kills itself (module state unknown) and the next tab starts clean
- creator load failure also kills the scope (see loadModelShared catch), so no tab joins a broken instance

The scope bundle does NOT contain the big generated code strings. The tab ships them at module-init time via `resources.jsPath.code` (emscripten) and `resources.llamaCppCode` (llama-cpp.js worker code). The build aliases `workers-code/generated` and `debug` to stubs (scripts/shared-worker-scope-stubs), final scope bundle is ~18KB minified.

The in-scope fake Worker (makeInScopeWorker): llama-cpp.js expects dedicated-worker globals. We define `self.postMessage`, run the code via importScripts on a blob URL, then capture `self.onmessage`. IMPORTANT: both delivery directions are deferred by a microtask; synchronous delivery breaks ProxyToWorker task ordering.

## Files

- `src/worker-shared.ts` - tab side: ProxyToSharedWorker, discovery/registry/locks, heartbeat, isSharedWorkerSupported
- `src/workers-code/shared-worker-scope.ts` - scope side: rpc server, state machine, in-scope worker shim
- `src/wllama.ts` - `sharedWorker` config flag; `loadModelShared()`; `initModules()` + `setLoadedState()` extracted from the old loadModel tail (both paths share them verbatim); model identity = joined `name:size` of prepared blobs, first model wins
- `src/worker.ts` - 4-line change: optional `workerFactory` ctor param, `resources.llamaCppCode` fallback
- `src/index.ts` - exports isSharedWorkerSupported
- `scripts/build_shared_worker_scope.mjs` + `scripts/shared-worker-scope-stubs/` - esbuild of the scope bundle with stub aliasing
- `scripts/build_worker.sh` - calls the above, embeds output as `SHARED_WORKER_SCOPE_CODE` in `src/workers-code/generated.ts`
- `src/worker-shared.test.ts` - vitest suite (6 tests)

REGEN RULE: any change to code reachable from shared-worker-scope.ts (worker.ts, glue, utils, the scope file itself) requires `npm run build:worker` to refresh SHARED_WORKER_SCOPE_CODE in generated.ts. Nothing enforces this yet, see future work.

## Tests

- `AUTO=1 npx vitest run src/worker-shared.test.ts` - in-repo suite. Runs in one page with two Wllama instances (each connect() is a separate port, so instance 2 exercises the true joiner path): create+load, hydrate with equal context info, concurrent completions, different-model rejection, exit-detach semantics.
- Multi-tab scenarios (creator tab closes, cold restart from stale registry, simultaneous cold-open race) cannot run inside vitest browser mode. They were validated with standalone playwright scripts driving real tabs during development (all passed), but those scripts lived in a session scratchpad and are gone. If needed again, rebuild from this doc: serve a page that calls the public API, drive 2-3 tabs, kill/reopen them. The scenarios and expected outcomes are listed above.

## Known limitations (v1, by design)

- Single-thread CPU inside the scope (browser limitation, no SAB). WebGPU unaffected.
- One model per origin at a time. Different model -> load_error "another model...".
- First loader's params win (n_ctx, n_gpu_layers, ...). Later tabs get the creator's context config. Only `seed` and `default_template_kwargs` stay per-tab.
- Model dies when the last tab closes (OPFS download cache survives, so reload skips the network).
- Firefox: works only via compat build path in theory (no JSPI in FF SharedWorker either), NOT tested. Safari: no Worker-in-SW until Safari 27, the feature-detect falls back to dedicated worker.
- A completion whose tab dies mid-generation leaves its server slot busy until the scope dies (no orphan-request cancellation yet).

## Future work ideas

- Map tabs to dedicated n_parallel sequences for stronger isolation.
- Orphaned request cleanup: scope tracks req_id -> portId, cancels requests whose port got pruned.
- CI guard that SHARED_WORKER_SCOPE_CODE in generated.ts is up to date with src.
- Multi-tab e2e in CI (playwright test project outside vitest).
- Expose an event so the app knows whether it hydrated vs loaded (for UX like "model already running in another tab").
