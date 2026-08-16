import { test, expect } from 'vitest';
import { Wllama } from './wllama';
import { ProxyToSharedWorker, isSharedWorkerSupported } from './worker-shared';

const CONFIG_PATHS = {
  default: '/src/wasm/wllama.wasm',
};

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

const OTHER_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf';

const createWllama = () => {
  const w = new Wllama(CONFIG_PATHS, {
    sharedWorker: true,
    suppressNativeLog: true,
  });
  w.setCompat(null);
  return w;
};

// connect() creates the scope when none exists, so this works as reset in both cases
const destroySharedScope = async () => {
  try {
    const probe = new ProxyToSharedWorker(
      { wasmPath: 'unused', compat: false },
      true,
      console
    );
    await probe.connect();
    await probe.destroy();
  } catch (e) {
    // scope may already be gone
  }
  await new Promise((r) => setTimeout(r, 500));
};

// both instances live in the same page, each connect() is a separate port, so wllamaB goes through the same joiner path as a second tab
let wllamaA: Wllama;
let wllamaB: Wllama;

test.sequential('shared worker is supported in test browser', async () => {
  expect(isSharedWorkerSupported()).toBe(true);
  await destroySharedScope();
});

test.sequential('first instance creates the scope and loads', async () => {
  wllamaA = createWllama();
  await wllamaA.loadModelFromUrl(TINY_MODEL, { n_ctx: 1024 });
  expect(wllamaA.isModelLoaded()).toBe(true);
  expect(wllamaA.isMultithread()).toBe(false); // no SharedArrayBuffer in a SharedWorker
  const res = await wllamaA.createCompletion({
    prompt: 'Once upon a time',
    max_tokens: 10,
  });
  expect(res.choices[0].text.length).toBeGreaterThan(0);
});

test.sequential('second instance attaches and hydrates', async () => {
  wllamaB = createWllama();
  await wllamaB.loadModelFromUrl(TINY_MODEL, { n_ctx: 1024 });
  expect(wllamaB.isModelLoaded()).toBe(true);
  expect(wllamaB.getModelMetadata()).toEqual(wllamaA.getModelMetadata());
  expect(wllamaB.getLoadedContextInfo()).toEqual(
    wllamaA.getLoadedContextInfo()
  );
  const res = await wllamaB.createCompletion({
    prompt: 'The little girl',
    max_tokens: 10,
  });
  expect(res.choices[0].text.length).toBeGreaterThan(0);
});

test.sequential('concurrent completions from both instances', async () => {
  const [r1, r2] = await Promise.all([
    wllamaA.createCompletion({ prompt: 'The cat', max_tokens: 16 }),
    wllamaB.createCompletion({ prompt: 'The dog', max_tokens: 16 }),
  ]);
  expect(r1.choices[0].text.length).toBeGreaterThan(0);
  expect(r2.choices[0].text.length).toBeGreaterThan(0);
});

test.sequential('loading a different model is rejected', async () => {
  const wllamaC = createWllama();
  await expect(
    wllamaC.loadModelFromUrl(OTHER_MODEL, { n_ctx: 512 })
  ).rejects.toThrow(/another model/);
});

test.sequential('exit() detaches one instance, the other keeps working', async () => {
  await wllamaA.exit();
  const res = await wllamaB.createCompletion({
    prompt: 'One day',
    max_tokens: 10,
  });
  expect(res.choices[0].text.length).toBeGreaterThan(0);
  await wllamaB.exit();
  await destroySharedScope();
});
