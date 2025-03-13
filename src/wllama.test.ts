import { test, expect } from 'vitest';
import { Wllama, WllamaChatMessage } from './wllama';

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': '/src/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': '/src/multi-thread/wllama.wasm',
};

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

const SPLIT_MODEL =
  'https://huggingface.co/ngxson/tinyllama_split_test/resolve/main/stories15M-q8_0-00001-of-00003.gguf';

const EMBD_MODEL = TINY_MODEL; // for better speed

test.sequential('loads single model file', async () => {
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

test.sequential('loads single model file from HF', async () => {
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

test.sequential('loads single thread model', async () => {
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

test.sequential('loads model with progress callback', async () => {
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

test.sequential('loads split model files', async () => {
  const wllama = new Wllama(CONFIG_PATHS, {
    parallelDownloads: 5,
  });

  await wllama.loadModelFromUrl(SPLIT_MODEL, {
    n_ctx: 1024,
  });

  expect(wllama.isModelLoaded()).toBe(true);
  await wllama.exit();
});

test.sequential('tokenizes and detokenizes text', async () => {
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

test.sequential('tokenize a long text', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
  });

  const text = 'hello '.repeat(1e4);
  const tokens = await wllama.tokenize(text);
  expect(tokens.length).toBeGreaterThan(10);

  await wllama.exit();
});

test.sequential('generates completion', async () => {
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

test.sequential('abort signal', async () => {
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
  const abortController = new AbortController();
  const stream = await wllama.createCompletion(prompt, {
    nPredict: 10,
    sampling: config,
    stream: true,
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

test.sequential('gets logits', async () => {
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

test.sequential('generates embeddings', async () => {
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

test.sequential('allowOffline', async () => {
  const wllama = new Wllama(CONFIG_PATHS, {
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

test.sequential('formatChat', async () => {
  const wllama = new Wllama(CONFIG_PATHS, {
    allowOffline: true,
  });

  await wllama.loadModelFromUrl(TINY_MODEL);
  expect(wllama.isModelLoaded()).toBe(true);
  const messages: WllamaChatMessage[] = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Hi!' },
    { role: 'assistant', content: 'Hello!' },
    { role: 'user', content: 'How are you?' },
  ];

  const formatted = await wllama.formatChat(messages, false);
  expect(formatted).toBe(
    '<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n'
  );

  const formatted1 = await wllama.formatChat(messages, true);
  expect(formatted1).toBe(
    '<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n'
  );

  const formatted2 = await wllama.formatChat(messages, true, 'zephyr');
  expect(formatted2).toBe(
    '<|system|>\nYou are helpful.<|endoftext|>\n<|user|>\nHi!<|endoftext|>\n<|assistant|>\nHello!<|endoftext|>\n<|user|>\nHow are you?<|endoftext|>\n<|assistant|>\n'
  );

  await wllama.exit();
});

test.sequential('generates chat completion', async () => {
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

  const messages: WllamaChatMessage[] = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Hi!' },
    { role: 'assistant', content: 'Hello!' },
    { role: 'user', content: 'How are you?' },
  ];
  const completion = await wllama.createChatCompletion(messages, {
    nPredict: 10,
    sampling: config,
  });

  expect(completion).toBeDefined();
  expect(completion).toMatch(/(Sudden|big|scary)+/);
  expect(completion.length).toBeGreaterThan(10);

  await wllama.exit();
});

test.sequential('generates chat completion using async iterator', async () => {
  const wllama = new Wllama(CONFIG_PATHS);

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
    seed: 42,
  });

  const messages: WllamaChatMessage[] = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Hi!' },
    { role: 'assistant', content: 'Hello!' },
    { role: 'user', content: 'How are you?' },
  ];
  const stream = await wllama.createChatCompletion(messages, {
    nPredict: 10,
    sampling: {
      temp: 0.0,
    },
    stream: true,
  });

  let finalTokens: number[] = [];
  let finalText = '';
  for await (const chunk of stream) {
    expect(chunk).toBeDefined();
    expect(chunk.token).toBeGreaterThan(0);
    expect(chunk.piece).toBeDefined();
    expect(chunk.piece.length).toBeGreaterThan(0);
    expect(chunk.currentText).toBeDefined();
    expect(chunk.currentText.length).toBeGreaterThan(0);
    finalTokens.push(chunk.token);
    finalText = chunk.currentText;
  }

  const detokenized = await wllama.detokenize(finalTokens, true);
  expect(finalText.length).toBeGreaterThan(10);
  expect(finalText).toMatch(/(Sudden|big|scary)+/);
  expect(detokenized).toBe(finalText);

  await wllama.exit();
});

test.sequential('cleans up resources', async () => {
  const wllama = new Wllama(CONFIG_PATHS);
  await wllama.loadModelFromUrl(TINY_MODEL);
  expect(wllama.isModelLoaded()).toBe(true);
  await wllama.exit();
  await expect(wllama.tokenize('test')).rejects.toThrow();

  // Double check that the model is really unloaded
  expect(wllama.isModelLoaded()).toBe(false);
});
