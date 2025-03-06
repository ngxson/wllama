import { test, expect } from 'vitest';
import { Wllama as WllamaMJS } from '../esm/index.js';
import { Wllama as WllamaMJSMinified } from '../esm/index.min.js';

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': '/src/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': '/src/multi-thread/wllama.wasm',
};

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

const testFunc = async (wllama: WllamaMJS) => {
  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
  });

  const config = {
    seed: 42,
    temp: 0.0,
    top_p: 0.95,
    top_k: 40,
  };

  await wllama.samplingInit(config);

  const prompt = 'Once upon a time';
  const completion = await wllama.createCompletion(prompt, {
    nPredict: 10,
    sampling: config,
  });

  expect(completion).toBeDefined();
  expect(completion).toMatch(/(there|little|girl|Lily)+/);
  expect(completion.length).toBeGreaterThan(10);

  await wllama.exit();
};

test.sequential('(mjs) generates completion', async () => {
  const wllama = new WllamaMJS(CONFIG_PATHS);
  await testFunc(wllama);
});

test.sequential('(mjs/minified) generates completion', async () => {
  const wllama = new WllamaMJSMinified(CONFIG_PATHS);
  await testFunc(wllama as unknown as WllamaMJS);
});
