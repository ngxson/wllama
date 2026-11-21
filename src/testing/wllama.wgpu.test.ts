import { test, expect } from 'vitest';
import { Wllama } from '../wllama';
import { LogLevel } from '../types/types';

const CONFIG_PATHS = {
  default: '/src/wasm/wllama.wasm',
};

// TODO: enable compat mode in tests once test infrastructure supports Safari/asyncify
const createWllama = (): Wllama => {
  const w = new Wllama(CONFIG_PATHS);
  w.setCompat(null);
  return w;
};

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

test('WebGPU is supported in this browser', () => {
  const wllama = createWllama();
  expect(wllama.isSupportWebGPU()).toBe(true);
});

test.sequential('loads model with WebGPU', async () => {
  const wllama = createWllama();

  expect(wllama.isSupportWebGPU()).toBe(true);

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
    n_gpu_layers: 99999,
  });

  expect(wllama.isModelLoaded()).toBe(true);
  expect(wllama.getModelMetadata()).toBeDefined();

  await wllama.exit();
});

test.sequential('parallel completions with WebGPU', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
    n_gpu_layers: 99999,
  });

  const prompts = [
    'Once upon a time',
    'The little girl said',
    'One day, a boy named',
  ];
  const results = await Promise.all(
    prompts.map((prompt) =>
      wllama.createCompletion({
        prompt,
        max_tokens: 10,
        temperature: 0.0,
        seed: 42,
      })
    )
  );

  expect(results.length).toBe(prompts.length);
  for (const res of results) {
    expect(res).toBeDefined();
    expect(res.choices[0].text.length).toBeGreaterThan(0);
  }

  // greedy + fixed seed: serial reruns must produce identical text
  for (let i = 0; i < prompts.length; i++) {
    const serial = await wllama.createCompletion({
      prompt: prompts[i],
      max_tokens: 10,
      temperature: 0.0,
      seed: 42,
    });
    expect(results[i].choices[0].text).toBe(serial.choices[0].text);
  }

  await wllama.exit();
});

test.sequential('generates completion with WebGPU', async () => {
  const wllama = createWllama();

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

// skipped queue waits (cpp/wllama-patch.cpp) must not read stale logits
// few tokens only: CPU and GPU results differ a bit, so long outputs can diverge
test.sequential('WebGPU completion matches CPU', async () => {
  const complete = async (n_gpu_layers: number) => {
    const wllama = createWllama();
    await wllama.loadModelFromUrl(TINY_MODEL, { n_ctx: 1024, n_gpu_layers });
    const res = await wllama.createCompletion({
      prompt: 'Once upon a time',
      max_tokens: 8,
      temperature: 0.0,
      seed: 42,
    });
    await wllama.exit();
    return res.choices[0].text;
  };

  const gpuText = await complete(99999);
  expect(gpuText.length).toBeGreaterThan(0);
  expect(gpuText).toBe(await complete(0));
});

test.sequential(
  'token embedding on GPU or CPU gives the same text',
  async () => {
    const complete = async (token_embd_on_gpu: boolean) => {
      const logs: string[] = [];
      const keep = (...args: any[]) => logs.push(args.join(' '));
      const wllama = new Wllama(CONFIG_PATHS, {
        logger: { debug: keep, log: keep, warn: keep, error: keep },
      });
      wllama.setCompat(null);
      await wllama.loadModelFromUrl(TINY_MODEL, {
        n_ctx: 1024,
        token_embd_on_gpu,
        log_level: LogLevel.DEBUG, // the override is logged at debug level
      });
      const res = await wllama.createCompletion({
        prompt: 'Once upon a time',
        max_tokens: 8,
        temperature: 0.0,
        seed: 42,
      });
      await wllama.exit();
      const overridden = logs.some((l) =>
        /token_embd\.weight.*overridden to WebGPU/.test(l)
      );
      return { text: res.choices[0].text, overridden };
    };

    const onGpu = await complete(true);
    const onCpu = await complete(false);
    expect(onGpu.overridden).toBe(true);
    expect(onCpu.overridden).toBe(false);
    expect(onGpu.text).toBe(onCpu.text);
  }
);
