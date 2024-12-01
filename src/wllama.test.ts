import { test, expect } from 'vitest';
import { Wllama } from './wllama';

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': '/src/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': '/src/multi-thread/wllama.wasm',
};

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

const SPLIT_MODEL =
  'https://huggingface.co/ngxson/tinyllama_split_test/resolve/main/stories15M-q8_0-00001-of-00003.gguf';

const EMBD_MODEL = TINY_MODEL; // for better speed

test('loads single model file', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

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

test('loads single model file from HF', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

  await wllama.loadModelFromHF(
    'ggml-org/models',
    'tinyllamas/stories15M-q4_0.gguf',
    {
      n_ctx: 1024,
      n_threads: 2,
    }
  );

  expect(wllama.isModelLoaded()).toBe(true);
  await wllama.exit();
});

test('loads single thread model', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
    n_threads: 1,
  });

  expect(wllama.isModelLoaded()).toBe(true);
  expect(wllama.isMultithread()).toBe(false);

  const completion = await wllama.createCompletion('Hello', { nPredict: 10 });
  expect(completion).toBeDefined();
  expect(completion.length).toBeGreaterThan(10);
  await wllama.exit();
});

test('loads model with progress callback', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

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

test('loads split model files', async () => {
  const wllama = new Wllama(CONFIG_PATHS, {
    parallelDownloads: 5,
  });

  await wllama.loadModelFromUrl(SPLIT_MODEL, {
    n_ctx: 1024,
  });

  expect(wllama.isModelLoaded()).toBe(true);
  await wllama.exit();
});

test('tokenizes and detokenizes text', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
  });

  const text =
    'Once Upon a Time is an American fantasy adventure drama television series';
  const tokens = await wllama.tokenize(text);
  expect(tokens.length).toBeGreaterThan(10);

  const detokenized = await wllama.detokenize(tokens);
  expect(detokenized.byteLength).toBeGreaterThan(10);

  const decodedText = new TextDecoder().decode(detokenized);
  expect(decodedText.trim()).toBe(text);

  await wllama.exit();
});

test('generates completion', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

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
});

test('gets logits', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
  });

  await wllama.samplingInit({});

  const logits = await wllama.getLogits(10);
  expect(logits.length).toBe(10);
  expect(logits[0]).toHaveProperty('token');
  expect(logits[0]).toHaveProperty('p');
  expect(logits[0].token).toBeGreaterThan(0);
  // expect(logits[0].p).toBeGreaterThan(0.5); // FIXME

  await wllama.exit();
});

test('generates embeddings', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

  await wllama.loadModelFromUrl(EMBD_MODEL, {
    n_ctx: 1024,
    embeddings: true,
  });

  expect(wllama.isModelLoaded()).toBe(true);

  const text = 'This is a test sentence';
  const embedding = await wllama.createEmbedding(text);

  expect(embedding).toBeDefined();
  expect(Array.isArray(embedding)).toBe(true);
  expect(embedding.length).toBeGreaterThan(0);
  expect(typeof embedding[0]).toBe('number');
  for (const e of embedding) {
    expect(typeof e).toBe('number');
    expect(e).toBeLessThan(1);
  }

  // make sure the vector is normalized
  const normVec = Math.sqrt(embedding.reduce((acc, v) => acc + v * v, 0));
  expect(Math.abs(normVec - 1)).toBeLessThan(1e-6);

  // slightly different text should have different embedding
  const embedding2 = await wllama.createEmbedding(text + ' ');
  const cosineDist = embedding.reduce(
    (acc, v, i) => acc + v * embedding2[i],
    0
  );
  expect(cosineDist).toBeGreaterThan(1 - 0.05);
  expect(cosineDist).toBeLessThan(1);

  await wllama.exit();
});

test('cleans up resources', async () => {
  const wllama = new Wllama(CONFIG_PATHS);
  await wllama.loadModelFromUrl(TINY_MODEL);
  expect(wllama.isModelLoaded()).toBe(true);
  await wllama.exit();
  await expect(wllama.tokenize('test')).rejects.toThrow();

  // Double check that the model is really unloaded
  expect(wllama.isModelLoaded()).toBe(false);
});
