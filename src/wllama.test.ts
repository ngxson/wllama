import { test, expect } from 'vitest';
import { Wllama, type WllamaConfig } from './wllama';

const CONFIG_PATHS = {
  default: '/src/wasm/wllama.wasm',
};

// TODO: enable compat mode in tests once test infrastructure supports Safari/asyncify
const createWllama = (config = CONFIG_PATHS, options: WllamaConfig = {}) => {
  const w = new Wllama(config, options);
  w.setCompat(null);
  return w;
};

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

const SPLIT_MODEL =
  'https://huggingface.co/ngxson/tinyllama_split_test/resolve/main/stories15M-q8_0-00001-of-00003.gguf';

const EMBD_MODEL = TINY_MODEL; // for better speed

test.sequential('loads single model file', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
    n_threads: 2,
  });

  expect(wllama.isModelLoaded()).toBe(true);
  expect(wllama.getModelMetadata()).toBeDefined();
  expect(wllama.getModelMetadata().hparams).toBeDefined();
  expect(wllama.isMultithread()).toBe(true);

  const metadata = wllama.getModelMetadata();
  expect(metadata.hparams).toBeDefined();
  expect(metadata.meta).toBeDefined();
  await wllama.exit();
});

test.sequential('loads single model file from HF', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromHF(
    { repo: 'ggml-org/models', file: 'tinyllamas/stories15M-q4_0.gguf' },
    {
      n_ctx: 1024,
      n_threads: 2,
    }
  );

  expect(wllama.isModelLoaded()).toBe(true);
  await wllama.exit();
});

test.sequential('loads single thread model', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
    n_threads: 1,
  });

  expect(wllama.isModelLoaded()).toBe(true);
  expect(wllama.isMultithread()).toBe(false);

  const res = await wllama.createCompletion({
    prompt: 'Hello',
    max_tokens: 10,
  });
  expect(res).toBeDefined();
  expect(res.choices[0].text.length).toBeGreaterThan(0);
  await wllama.exit();
});

test.sequential('loads model with progress callback', async () => {
  const wllama = createWllama();

  let progressCalled = false;
  let lastLoaded = 0;
  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
    progressCallback: ({ loaded, total }) => {
      expect(loaded).toBeGreaterThan(0);
      expect(total).toBeGreaterThan(0);
      expect(loaded).toBeLessThanOrEqual(total);
      expect(loaded).toBeGreaterThanOrEqual(lastLoaded);
      progressCalled = true;
      lastLoaded = loaded;
    },
  });

  expect(progressCalled).toBe(true);
  expect(wllama.isModelLoaded()).toBe(true);
  await wllama.exit();
});

test.sequential('loads split model files', async () => {
  const wllama = createWllama(CONFIG_PATHS, {
    parallelDownloads: 5,
  });

  await wllama.loadModelFromUrl(SPLIT_MODEL, {
    n_ctx: 1024,
  });

  expect(wllama.isModelLoaded()).toBe(true);
  await wllama.exit();
});

test.sequential('generates completion', async () => {
  const wllama = createWllama();

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
});

test.sequential('abort signal', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
  });

  const abortController = new AbortController();
  const stream = await wllama.createCompletion({
    prompt: 'Once upon a time',
    max_tokens: 10,
    temperature: 0.0,
    top_p: 0.95,
    top_k: 40,
    seed: 42,
    stream: true,
    onData: () => {},
    abortSignal: abortController.signal,
  });

  let i = 0;
  for await (const _ of stream) {
    if (i === 2) {
      abortController.abort();
    }
    i++;
  }

  expect(i).toBe(4);

  await wllama.exit();
});

test.sequential('generates embeddings', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(EMBD_MODEL, {
    n_ctx: 1024,
    embeddings: true,
  });

  expect(wllama.isModelLoaded()).toBe(true);

  const text = 'This is a test sentence';
  const res = await wllama.createEmbedding({ input: text });

  expect(res).toBeDefined();
  const embedding = res.data[0].embedding as number[];
  expect(Array.isArray(embedding)).toBe(true);
  expect(embedding.length).toBeGreaterThan(0);
  for (const e of embedding) {
    expect(typeof e).toBe('number');
  }

  // slightly different text should have high cosine similarity
  const res2 = await wllama.createEmbedding({ input: text + ' ' });
  const embedding2 = res2.data[0].embedding as number[];
  const dot = embedding.reduce((acc, v, i) => acc + v * embedding2[i], 0);
  const norm1 = Math.sqrt(embedding.reduce((acc, v) => acc + v * v, 0));
  const norm2 = Math.sqrt(embedding2.reduce((acc, v) => acc + v * v, 0));
  const cosineSim = dot / (norm1 * norm2);
  expect(cosineSim).toBeGreaterThan(1 - 0.05);
  expect(cosineSim).toBeLessThan(1);

  await wllama.exit();
});

test.sequential('allowOffline', async () => {
  const wllama = createWllama(CONFIG_PATHS, {
    allowOffline: true,
  });

  // Mock fetch to simulate offline
  const origFetch = window.fetch;
  window.fetch = () => Promise.reject(new Error('offline'));

  try {
    await wllama.loadModelFromUrl(TINY_MODEL);
    expect(wllama.isModelLoaded()).toBe(true);
    await wllama.exit();
  } catch (e) {
    window.fetch = origFetch;
    throw e;
  } finally {
    window.fetch = origFetch;
  }
});

test.sequential('generates chat completion', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
  });

  const res = await wllama.createChatCompletion({
    messages: [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Hi!' },
      { role: 'assistant', content: 'Hello!' },
      { role: 'user', content: 'How are you?' },
    ],
    max_tokens: 10,
    temperature: 0.0,
    top_p: 0.95,
    top_k: 40,
    seed: 42,
  });

  const text = res.choices[0].message.content as string;
  expect(text).toBeDefined();
  expect(text).toMatch(/(Sudden|big|scary)+/);
  expect(text.length).toBeGreaterThan(10);

  await wllama.exit();
});

test.sequential('generates chat completion using async iterator', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
    seed: 42,
  });

  const stream = await wllama.createChatCompletion({
    messages: [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Hi!' },
      { role: 'assistant', content: 'Hello!' },
      { role: 'user', content: 'How are you?' },
    ],
    max_tokens: 10,
    temperature: 0.0,
    stream: true,
    onData: () => {},
  });

  let finalText = '';
  for await (const chunk of stream) {
    expect(chunk).toBeDefined();
    expect(chunk.object).toBe('chat.completion.chunk');
    const delta = chunk.choices[0].delta;
    if (delta.content) {
      finalText += delta.content;
    }
  }

  expect(finalText.length).toBeGreaterThan(10);
  expect(finalText).toMatch(/(Sudden|big|scary)+/);

  await wllama.exit();
});

test.sequential('cleans up resources', async () => {
  const wllama = createWllama();
  await wllama.loadModelFromUrl(TINY_MODEL);
  expect(wllama.isModelLoaded()).toBe(true);
  await wllama.exit();
  await expect(
    wllama.createCompletion({ prompt: 'test', max_tokens: 1 })
  ).rejects.toThrow();

  // Double check that the model is really unloaded
  expect(wllama.isModelLoaded()).toBe(false);
});
