import { test, expect } from 'vitest';
import { Wllama as WllamaMJS } from '../esm/index.js';
import { Wllama as WllamaMJSMinified } from '../esm/index.min.js';

const CONFIG_PATHS = {
  default: '/src/wasm/wllama.wasm',
};

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

const testFunc = async (wllama: WllamaMJS) => {
  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
  });

  const res = await wllama.createCompletion({
    prompt: 'Once upon a time',
    max_tokens: 10,
    temperature: 0.0,
    top_p: 0.95,
    top_k: 40,
    seed: 42,
  });

  expect(res).toBeDefined();
  expect(res.choices[0].text).toMatch(/(there|little|girl|Lily)+/);
  expect(res.choices[0].text.length).toBeGreaterThan(10);

  await wllama.exit();
};

// TODO: enable compat mode in tests once test infrastructure supports Safari/asyncify
test.sequential('(mjs) generates completion', async () => {
  const wllama = new WllamaMJS(CONFIG_PATHS);
  wllama.setCompat(null);
  await testFunc(wllama);
});

test.sequential('(mjs/minified) generates completion', async () => {
  const wllama = new WllamaMJSMinified(CONFIG_PATHS);
  wllama.setCompat(null);
  await testFunc(wllama as unknown as WllamaMJS);
});
