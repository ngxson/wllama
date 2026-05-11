import { test, expect } from 'vitest';
import { Wllama } from './wllama';

const CONFIG_PATHS = {
  default: '/src/wasm/wllama.wasm',
};

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

test('WebGPU is supported in this browser', () => {
  const wllama = new Wllama(CONFIG_PATHS);
  expect(wllama.isSupportWebGPU()).toBe(true);
});

test.sequential('loads model with WebGPU', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

  expect(wllama.isSupportWebGPU()).toBe(true);

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
    n_gpu_layers: 99999,
  });

  expect(wllama.isModelLoaded()).toBe(true);
  expect(wllama.getModelMetadata()).toBeDefined();

  await wllama.exit();
});

test.sequential('generates completion with WebGPU', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

  expect(wllama.isSupportWebGPU()).toBe(true);

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
    n_gpu_layers: 99999,
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
  expect(res.choices[0].text.length).toBeGreaterThan(0);

  await wllama.exit();
});
