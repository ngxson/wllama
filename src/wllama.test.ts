import { test, expect, beforeEach, vi } from 'vitest';

declare const __GITHUB_CI__: boolean;

// Add a small delay before each test on GitHub CI to avoid HuggingFace rate limits.
// typeof guard handles the case where vitest define is not configured.
if (typeof __GITHUB_CI__ !== 'undefined' && __GITHUB_CI__) {
  beforeEach(async () => {
    await new Promise((resolve) => setTimeout(resolve, 100));
  });
}
import {
  Wllama,
  decodeRealtimeAudio,
  encodeRealtimeAudio,
  type WllamaConfig,
  type WllamaRealtimeSession,
} from './wllama';
import type { RealtimeEvent } from './types/oai-compat';

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

const VISION_MODEL = {
  url: 'https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/tinygemma3-Q8_0.gguf',
  mmprojUrl:
    'https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/mmproj-tinygemma3.gguf',
};

const cosineSim = (a: number[], b: number[]) => {
  const dot = a.reduce((acc, v, i) => acc + v * b[i], 0);
  const norm = (v: number[]) => Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
  return dot / (norm(a) * norm(b));
};

const createTestImage = async (color: string) => {
  const canvas = new OffscreenCanvas(32, 32);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 32, 32);
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  return await blob.arrayBuffer();
};

const RERANK_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/jina-reranker-v1-tiny-en/ggml-model-f16.gguf';

const SYSTEMONE_MODEL =
  'https://huggingface.co/ggml-org/tinyopenjev-for-testing-gguf/resolve/main/tinyopenjev-for-testing-Q8_0.gguf';

// its output is gibberish, tests can only check the format and the flow, not the content
const AUDIO_MODEL_REPO =
  'https://huggingface.co/ggml-org/tinyLFM3-Omni-for-testing-gguf/resolve/main';
const AUDIO_MODEL = {
  url: `${AUDIO_MODEL_REPO}/tinyLFM3-Omni-for-testing-Q2_K.gguf`,
  mmprojUrl: `${AUDIO_MODEL_REPO}/mmproj-tinyLFM3-Omni-for-testing-Q2_K.gguf`,
};

const loadAudioModel = async (n_threads = 2, n_gpu_layers?: number) => {
  const wllama = createWllama();
  await wllama.loadModelFromUrl(AUDIO_MODEL, {
    n_ctx: 2048,
    n_threads,
    n_gpu_layers,
  });
  return wllama;
};

const createTestPcm = (seconds = 1, rate = 16000, freq = 440) => {
  const samples = new Float32Array(seconds * rate);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = 0.3 * Math.sin((2 * Math.PI * freq * i) / rate);
  }
  return samples;
};

const createTestWav = (seconds = 1, rate = 16000, freq = 440) => {
  const samples = createTestPcm(seconds, rate, freq);
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVEfmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((v, i) => view.setInt16(44 + i * 2, v * 32767, true));
  return view.buffer;
};

const readWavHeader = (audio: ArrayBuffer) => {
  const view = new DataView(audio);
  const str = (offset: number) =>
    String.fromCharCode(...new Uint8Array(audio, offset, 4));
  return {
    riff: str(0),
    wave: str(8),
    channels: view.getUint16(22, true),
    rate: view.getUint32(24, true),
    bits: view.getUint16(34, true),
  };
};

// reads events up to and including the one that matches
const readUntil = async (
  session: WllamaRealtimeSession,
  isLast: (ev: RealtimeEvent) => boolean
) => {
  const events: RealtimeEvent[] = [];
  for await (const ev of session.events()) {
    events.push(ev);
    if (ev.type === 'error' || isLast(ev)) {
      break;
    }
  }
  return events;
};

const countEvents = (events: RealtimeEvent[], type: string) =>
  events.filter((ev) => ev.type === type).length;

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

test.sequential('generates completions in parallel', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(TINY_MODEL, {
    n_ctx: 1024,
  });

  // concurrent requests must not interfere with each other (issue #261)
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
    abortSignal: abortController.signal,
  });

  let i = 0;
  try {
    for await (const _ of stream) {
      if (i === 2) {
        abortController.abort();
      }
      i++;
    }
  } catch (e) {
    expect((e as Error).name).toBe('AbortError');
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

  await expect(
    wllama.createEmbedding({ input: text, embd_normalize: 0.5 })
  ).rejects.toThrow('embd_normalize must be an integer');
  await expect(
    wllama.createEmbedding({ input: text, embd_normalize: -2 })
  ).rejects.toThrow('embd_normalize must be an integer');
  await expect(
    wllama.createEmbedding({ input: [{ type: 'text' } as any] })
  ).rejects.toThrow('Text input must contain a string');
  await expect(
    wllama.createEmbedding({ input: [{ type: 'text', text: '' }] })
  ).rejects.toThrow('Input content cannot be empty');
  await expect(
    wllama.createEmbedding({
      input: [{ type: 'image', data: new ArrayBuffer(8) }],
    })
  ).rejects.toThrow('Image input is not supported');

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

test.sequential('embeds more inputs than n_parallel', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(EMBD_MODEL, {
    n_ctx: 1024,
    embeddings: true,
    n_parallel: 2,
  });

  const input = ['one', 'two', 'three', 'four', 'five'];
  const res = await wllama.createEmbedding({ input });
  expect(res.data.map((d) => d.index)).toEqual([0, 1, 2, 3, 4]);

  const single = await wllama.createEmbedding({ input: input[3] });
  expect(
    cosineSim(
      res.data[3].embedding as number[],
      single.data[0].embedding as number[]
    )
  ).toBeGreaterThan(0.99);

  await wllama.exit();
});

test.sequential(
  'embeds images',
  async () => {
    const wllama = createWllama();

    await wllama.loadModelFromUrl(VISION_MODEL, {
      n_ctx: 1024,
      embeddings: true,
      pooling_type: 'last',
      n_parallel: 2,
      n_threads: 4,
    });

    const data = await createTestImage('red');
    const image = { type: 'image' as const, data };
    const text = { type: 'text' as const, text: 'a picture' };

    const ref = await wllama.createEmbedding({ input: [image, text] });
    const refEmbd = ref.data[0].embedding as number[];
    expect(ref.data.length).toBe(1);

    // a prompt that ends with an image must not abort
    const res = await wllama.createEmbedding({ input: [text, image] });
    expect((res.data[0].embedding as number[]).length).toBe(refEmbd.length);

    // more requests than slots: the extra ones wait for a free slot and keep their image
    const all = await Promise.all(
      [0, 1, 2, 3].map(() => wllama.createEmbedding({ input: [image, text] }))
    );
    for (const r of all) {
      expect(
        cosineSim(r.data[0].embedding as number[], refEmbd)
      ).toBeGreaterThan(0.99);
    }

    // one embedding per entry
    const batch = await wllama.createEmbedding({
      input: [[image, text], [{ type: 'text', text: 'no image here' }]],
    });
    expect(batch.data.length).toBe(2);
    expect(
      cosineSim(batch.data[0].embedding as number[], refEmbd)
    ).toBeGreaterThan(0.99);

    await wllama.exit();
  },
  120000
);

test.sequential('rejects images if pooling loses them', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(VISION_MODEL, {
    n_ctx: 1024,
    embeddings: true,
    pooling_type: 'mean',
  });

  const data = await createTestImage('red');
  await expect(
    wllama.createEmbedding({ input: [{ type: 'image', data }] })
  ).rejects.toThrow('requires pooling_type = last');

  await wllama.exit();
});

test.sequential('reranks documents', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(RERANK_MODEL, {
    embeddings: true,
    pooling_type: 'rank',
  });

  expect(wllama.isModelLoaded()).toBe(true);

  const query = 'What is machine learning?';
  const documents = [
    'Machine learning is a branch of artificial intelligence.',
    'The weather today is sunny and warm.',
    'Neural networks are used in deep learning.',
  ];

  const res = await wllama.createRerank({ query, documents });

  expect(res).toBeDefined();
  expect(res.results).toHaveLength(documents.length);
  for (const r of res.results) {
    expect(typeof r.index).toBe('number');
    expect(typeof r.relevance_score).toBe('number');
  }

  // results should be sorted highest score first
  for (let i = 0; i < res.results.length - 1; i++) {
    expect(res.results[i].relevance_score).toBeGreaterThanOrEqual(
      res.results[i + 1].relevance_score
    );
  }

  // the most relevant documents should outscore the other
  const weatherIdx = res.results.findIndex((r) => r.index === 1);
  expect(weatherIdx).toBeGreaterThan(0);

  await wllama.exit();
});

test.sequential('answers systemone questions', async () => {
  const wllama = createWllama();

  await wllama.loadModelFromUrl(SYSTEMONE_MODEL, {
    n_ctx: 4096,
  });

  const res = await wllama.createSystemOne({
    state: 'I was charged twice for my order last week and nobody has replied.',
    questions: {
      route: {
        type: 'choice',
        instructions: 'Which team should handle this?',
        criteria: {
          billing: 'payments and refunds',
          shipping: null,
          technical: null,
        },
      },
      urgency: {
        type: 'score',
        instructions: 'How urgent is this?',
        criteria: ['can wait', 'this week', 'today', 'right now'],
      },
      angry: {
        type: 'noul',
        instructions: 'Is the customer angry?',
      },
    },
  });

  expect(res.usage.input_tokens).toBeGreaterThan(0);
  expect(res.usage.output_tokens).toBe(0);
  expect(Object.keys(res.answers)).toEqual(['route', 'urgency', 'angry']);

  const sum = (p: Record<string, number>) =>
    Object.values(p).reduce((a, b) => a + b, 0);

  const route = res.answers.route;
  if (route.type !== 'choice') throw new Error('bad type');
  expect(Object.keys(route.probabilities)).toEqual([
    'billing',
    'shipping',
    'technical',
  ]);
  expect(sum(route.probabilities)).toBeCloseTo(1, 4);
  expect(route.confidence).toBeGreaterThanOrEqual(0);
  expect(route.confidence).toBeLessThanOrEqual(1);

  const urgency = res.answers.urgency;
  if (urgency.type !== 'score') throw new Error('bad type');
  expect(Object.keys(urgency.probabilities)).toEqual(['0', '1', '2', '3']);
  expect(sum(urgency.probabilities)).toBeCloseTo(1, 4);

  const angry = res.answers.angry;
  if (angry.type !== 'noul') throw new Error('bad type');
  expect(angry.noul).toBeGreaterThanOrEqual(0);
  expect(angry.noul).toBeLessThanOrEqual(1);

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

test('checks wasm version', async () => {
  const check = (version?: string, options: WllamaConfig = {}) =>
    (createWllama(CONFIG_PATHS, options) as any).checkWasmVersion(version);
  const expected = Wllama.getLibllamaVersion();
  const commit = expected.split('-').pop();

  await expect(check(expected)).resolves.toBeUndefined();
  // the build number is wrong in a shallow clone
  await expect(check(`b1-${commit}`)).resolves.toBeUndefined();

  const mismatch = { type: 'version_mismatch' };
  await expect(check('b1-0000000')).rejects.toMatchObject(mismatch);
  // a wasm file from before the version check
  await expect(check(undefined)).rejects.toMatchObject(mismatch);

  const warn = vi.fn();
  await expect(
    check('b1-0000000', {
      skipVersionCheck: true,
      logger: { ...console, warn },
    })
  ).resolves.toBeUndefined();
  expect(warn).toHaveBeenCalledWith(expect.stringContaining('b1-0000000'));
});

test.sequential('stack trace (abort)', async () => {
  const wllama = createWllama();
  await wllama.loadModelFromUrl(TINY_MODEL, {
    pooling_type: 'test_stack_trace_abort' as any,
  });
  expect(wllama.isModelLoaded()).toBe(true);

  const err1: unknown = await wllama
    .createCompletion({ prompt: 'test', max_tokens: 1 })
    .catch((e: unknown) => e);
  expect(err1).toBeInstanceOf(Error);
  expect((err1 as Error).name).toBe('RuntimeError');
  expect((err1 as Error).stack).toMatch(/__wrap_abort/);
  expect((err1 as Error).stack).toMatch(/server_response::send/);

  await wllama.exit();
});

// TODO @ngxson : this stucks on github CI but not on local run, investigate why and re-enable
test.skip('stack trace (OOB memory access)', async () => {
  const wllama = createWllama();
  await wllama.loadModelFromUrl(TINY_MODEL, {
    pooling_type: 'test_stack_trace_oob' as any,
    n_threads: 1, // multithread stucks on github CI but not on local run, why?
  });
  expect(wllama.isModelLoaded()).toBe(true);

  const err2: unknown = await wllama
    .createCompletion({ prompt: 'test', max_tokens: 1 })
    .catch((e: unknown) => e);
  expect(err2).toBeInstanceOf(Error);
  expect((err2 as Error).name).toBe('RuntimeError');
  expect((err2 as Error).stack).toMatch(/server_response::send/);

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

test.sequential(
  'reports audio capabilities',
  async () => {
    const wllama = await loadAudioModel();
    const info = wllama.getLoadedContextInfo();
    expect(info.has_image_input).toBe(true);
    expect(info.has_audio_input).toBe(true);
    expect(info.has_audio_output).toBe(true);
    expect(info.has_audio_stream).toBe(true);
    expect(info.has_live_asr).toBe(true);
    expect(info.audio_input_rate).toBe(16000);
    expect(info.audio_output_rate).toBe(24000);
    expect(info.audio_voices).toEqual(['female', 'male']);
    await wllama.exit();
  },
  120000
);

test.sequential(
  'a text model has no audio',
  async () => {
    const wllama = createWllama();
    await wllama.loadModelFromUrl(TINY_MODEL, { n_ctx: 256 });
    const info = wllama.getLoadedContextInfo();
    expect(info.has_audio_output).toBe(false);
    expect(info.has_audio_stream).toBe(false);
    expect(info.audio_voices).toEqual([]);
    await expect(wllama.createSpeech({ input: 'hello' })).rejects.toThrow(
      'does not support audio generation'
    );
    await expect(
      wllama.createTranscription({ file: createTestWav() })
    ).rejects.toThrow('Audio input is not supported');
    await expect(wllama.createRealtimeSession()).rejects.toThrow(
      'does not support realtime audio'
    );
    await wllama.exit();
  },
  60000
);

test.sequential(
  'generates speech',
  async () => {
    const wllama = await loadAudioModel();
    const params = { input: 'Hello world', n_predict: 16, seed: 42 };

    const res = await wllama.createSpeech(params);
    expect(res.sample_rate).toBe(24000);
    expect(res.audio.byteLength).toBeGreaterThan(44);
    expect(readWavHeader(res.audio)).toEqual({
      riff: 'RIFF',
      wave: 'WAVE',
      channels: 1,
      rate: 24000,
      bits: 16,
    });

    // same seed, same audio
    const again = await wllama.createSpeech(params);
    expect(new Uint8Array(again.audio)).toEqual(new Uint8Array(res.audio));

    const male = await wllama.createSpeech({ ...params, voice: 'male' });
    expect(new Uint8Array(male.audio)).not.toEqual(new Uint8Array(res.audio));

    await expect(
      wllama.createSpeech({ ...params, voice: 'nobody' })
    ).rejects.toThrow('Unknown voice');
    await expect(wllama.createSpeech({ input: '' })).rejects.toThrow(
      'non-empty'
    );
    await wllama.exit();
  },
  120000
);

test.sequential(
  'generates speech with a speaker reference',
  async () => {
    const wllama = await loadAudioModel();
    // LFM3-Omni ignores it, this only checks that the audio is decoded and passed
    const speaker_ref = createTestWav(1, 24000, 1000);
    const res = await wllama.createSpeech({
      input: 'Hello world',
      n_predict: 16,
      seed: 42,
      speaker_ref,
    });
    expect(readWavHeader(res.audio).riff).toBe('RIFF');
    expect(res.audio.byteLength).toBeGreaterThan(44);

    await expect(
      wllama.createSpeech({ input: 'Hello', speaker_ref, voice: 'male' })
    ).rejects.toThrow('cannot be used together');
    await expect(
      wllama.createSpeech({ input: 'Hello', speaker_ref: new ArrayBuffer(16) })
    ).rejects.toThrow('Failed to decode');
    await wllama.exit();
  },
  120000
);

test.sequential(
  'streams speech',
  async () => {
    const wllama = await loadAudioModel();
    const chunks = [];
    for await (const chunk of await wllama.createSpeech({
      input: 'Hello world',
      n_predict: 16,
      seed: 42,
      response_format: 'pcm',
      stream: true,
    })) {
      chunks.push(chunk);
    }
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.filter((c) => c.final).length).toBe(1);
    expect(chunks.at(-1)!.final).toBe(true);
    const nBytes = chunks.reduce((acc, c) => acc + c.audio.byteLength, 0);
    expect(nBytes).toBeGreaterThan(0);
    expect(nBytes % 4).toBe(0); // float32 samples

    // a reader that stops early must free the slot
    for (let i = 0; i < 6; i++) {
      for await (const chunk of await wllama.createSpeech({
        input: 'Hello world',
        response_format: 'pcm',
        stream: true,
      })) {
        expect(chunk.sample_rate).toBe(24000);
        break;
      }
    }
    const res = await wllama.createSpeech({ input: 'Hi', n_predict: 4 });
    expect(res.audio.byteLength).toBeGreaterThan(44);
    await wllama.exit();
  },
  120000
);

test.sequential(
  'stops speech on abort',
  async () => {
    const wllama = await loadAudioModel();
    const aborted = new AbortController();
    aborted.abort();
    const err = await wllama
      .createSpeech({ input: 'Hello', abortSignal: aborted.signal })
      .catch((e: unknown) => e);
    expect((err as Error).name).toBe('AbortError');

    // abort while the request is being read, then the slots must still be free
    for (let i = 0; i < 6; i++) {
      const abort = new AbortController();
      const stream = await wllama.createSpeech({
        input: 'Hello world',
        response_format: 'pcm',
        stream: true,
        abortSignal: abort.signal,
      });
      const err = await (async () => {
        for await (const _ of stream) {
          abort.abort();
        }
      })().catch((e: unknown) => e);
      expect((err as Error).name).toBe('AbortError');
    }
    const res = await wllama.createSpeech({ input: 'Hi', n_predict: 4 });
    expect(res.audio.byteLength).toBeGreaterThan(44);
    await wllama.exit();
  },
  120000
);

test.sequential(
  'transcribes an audio file',
  async () => {
    const wllama = await loadAudioModel();
    const res = await wllama.createTranscription({
      file: createTestWav(),
      language: 'en',
      max_tokens: 8,
      temperature: 0.5,
    });
    expect(res.type).toBe('transcript.text.done');
    expect(typeof res.text).toBe('string');
    expect(res.usage.input_tokens).toBeGreaterThan(0);
    expect(res.usage.output_tokens).toBe(8);

    const chunks = [];
    for await (const chunk of await wllama.createTranscription({
      file: createTestWav(),
      max_tokens: 8,
      stream: true,
    })) {
      chunks.push(chunk);
    }
    expect(chunks.at(-1)!.type).toBe('transcript.text.done');

    await expect(
      wllama.createTranscription({ file: new ArrayBuffer(16) })
    ).rejects.toThrow();
    await wllama.exit();
  },
  120000
);

// audio preprocessing used to start more threads than the pool has, and hang
test.sequential.each([1, 2, 4])(
  'takes audio input with %i thread(s)',
  async (n_threads) => {
    const wllama = await loadAudioModel(n_threads);
    const res = await wllama.createChatCompletion({
      messages: [
        {
          role: 'user',
          content: [{ type: 'audio', data: createTestWav() }],
        },
      ],
      max_tokens: 4,
    });
    expect(res.usage!.completion_tokens).toBe(4);

    const session = await wllama.createRealtimeSession();
    await session.send({
      type: 'session.update',
      session: { type: 'transcription', max_output_tokens: 4 },
    });
    await session.appendAudio(createTestPcm(1));
    await session.send({ type: 'input_audio_buffer.commit' });
    const events = await readUntil(session, (e) =>
      e.type.endsWith('input_audio_transcription.completed')
    );
    expect(events.filter((e) => e.type === 'error')).toEqual([]);
    expect(events.at(-1)!.type).toBe(
      'conversation.item.input_audio_transcription.completed'
    );
    await session.close();
    await wllama.exit();
  },
  120000
);

test('realtime audio codec is reversible', () => {
  const samples = createTestPcm(1);
  const decoded = decodeRealtimeAudio(encodeRealtimeAudio(samples));
  expect(decoded.length).toBe(samples.length);
  samples.forEach((v, i) => expect(decoded[i]).toBeCloseTo(v, 3));
});

test.sequential(
  'holds a realtime conversation',
  async () => {
    const wllama = await loadAudioModel();
    const session = await wllama.createRealtimeSession({ seed: 42 });

    let events = await readUntil(session, (e) => e.type === 'session.created');
    expect(events.at(-1)!.session.type).toBe('realtime');
    expect(events.at(-1)!.session.audio.input.format.rate).toBe(16000);
    expect(events.at(-1)!.session.audio.output.format.rate).toBe(24000);

    await session.send({
      type: 'session.update',
      session: { output_modalities: ['audio', 'text'], max_output_tokens: 24 },
    });
    await session.appendAudio(createTestPcm(1));
    await session.send({ type: 'input_audio_buffer.commit' });
    await session.send({ type: 'response.create' });
    events = await readUntil(session, (e) => e.type === 'response.done');
    expect(events.filter((e) => e.type === 'error')).toEqual([]);
    expect(countEvents(events, 'session.updated')).toBe(1);
    expect(countEvents(events, 'input_audio_buffer.committed')).toBe(1);
    expect(countEvents(events, 'response.created')).toBe(1);
    expect(countEvents(events, 'response.output_text.delta')).toBeGreaterThan(
      0
    );
    const audio = events.filter(
      (e) => e.type === 'response.output_audio.delta'
    );
    expect(audio.length).toBeGreaterThan(0);
    expect(decodeRealtimeAudio(audio[0].delta).length).toBeGreaterThan(0);
    expect(events.at(-1)!.response.status).toBe('completed');

    // next turn in the same session, with text only
    await session.send({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Hello' }],
      },
    });
    await session.send({
      type: 'response.create',
      response: { output_modalities: ['text'] },
    });
    events = await readUntil(session, (e) => e.type === 'response.done');
    expect(events.filter((e) => e.type === 'error')).toEqual([]);
    expect(countEvents(events, 'response.output_audio.delta')).toBe(0);
    expect(events.at(-1)!.response.status).toBe('completed');

    // protocol errors are events, the session stays open
    await session.send({ type: 'input_audio_buffer.commit' });
    events = await readUntil(session, (e) => e.type === 'error');
    expect(events.at(-1)!.error.type).toBe('input_audio_buffer_commit_empty');

    await session.close();
    await expect(session.send({ type: 'response.create' })).rejects.toThrow(
      'closed'
    );
    await wllama.exit();
  },
  120000
);

test.sequential(
  'cancels a realtime response',
  async () => {
    const wllama = await loadAudioModel();
    const session = await wllama.createRealtimeSession();
    await session.send({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Hello' }],
      },
    });
    await session.send({
      type: 'response.create',
      response: { output_modalities: ['text'] },
    });
    await readUntil(session, (e) => e.type === 'response.output_text.delta');
    await session.send({ type: 'response.cancel' });
    const events = await readUntil(session, (e) => e.type === 'response.done');
    expect(events.at(-1)!.response.status).toBe('cancelled');
    await session.close();
    await wllama.exit();
  },
  120000
);

test.sequential(
  'transcribes in a realtime session',
  async () => {
    const wllama = await loadAudioModel();
    const session = await wllama.createRealtimeSession({ seed: 42 });
    await session.send({
      type: 'session.update',
      session: { type: 'transcription', max_output_tokens: 16 },
    });

    // each commit is transcribed on its own
    for (let i = 0; i < 2; i++) {
      await session.appendAudio(createTestPcm(1));
      await session.send({ type: 'input_audio_buffer.commit' });
      const events = await readUntil(session, (e) =>
        e.type.endsWith('input_audio_transcription.completed')
      );
      expect(events.filter((e) => e.type === 'error')).toEqual([]);
      const deltas = events.filter((e) =>
        e.type.endsWith('input_audio_transcription.delta')
      );
      expect(deltas.length).toBeGreaterThan(0);
      expect(events.at(-1)!.transcript).toBe(
        deltas.map((e) => e.delta).join('')
      );
    }
    await session.close();
    await wllama.exit();
  },
  120000
);

test.sequential(
  'transcribes live',
  async () => {
    const wllama = await loadAudioModel();
    const session = await wllama.createRealtimeSession({
      intent: 'transcription',
      seed: 42,
    });

    let events = await readUntil(session, (e) => e.type === 'session.created');
    expect(events.at(-1)!.session.type).toBe('transcription');

    // a commit completes its item, more audio can follow
    const pcm = createTestPcm(2);
    await session.appendAudio(pcm);
    await session.send({ type: 'input_audio_buffer.commit' });
    events = await readUntil(session, (e) =>
      e.type.endsWith('input_audio_transcription.completed')
    );
    expect(events.filter((e) => e.type === 'error')).toEqual([]);
    expect(countEvents(events, 'input_audio_buffer.committed')).toBe(1);
    const first = events.at(-1)!;

    // a final commit ends the session
    await session.appendAudio(pcm);
    await session.send({ type: 'input_audio_buffer.commit', final: true });
    events = [];
    for await (const ev of session.events()) {
      events.push(ev);
    }
    expect(events.filter((e) => e.type === 'error')).toEqual([]);
    const last = events.at(-1)!;
    expect(last.type).toBe(
      'conversation.item.input_audio_transcription.completed'
    );
    expect(last.item_id).not.toBe(first.item_id);
    await expect(session.send({ type: 'response.create' })).rejects.toThrow(
      'closed'
    );
    await wllama.exit();
  },
  120000
);

test.sequential(
  'frees the slot of a closed realtime session',
  async () => {
    const wllama = await loadAudioModel();
    // more sessions than slots
    for (let i = 0; i < 6; i++) {
      const session = await wllama.createRealtimeSession();
      await session.appendAudio(createTestPcm(1));
      await session.close();
    }
    const res = await wllama.createChatCompletion({
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 4,
    });
    expect(res.usage!.completion_tokens).toBe(4);
    await wllama.exit();
  },
  120000
);

test.sequential(
  'runs other requests during a realtime session',
  async () => {
    // on WebGPU, 2 TTS at once change each other's sampled audio (still valid audio), so compare on CPU
    const wllama = await loadAudioModel(2, 0);
    const chat = () =>
      wllama.createChatCompletion({
        messages: [{ role: 'user', content: 'Hello' }],
        max_tokens: 8,
        temperature: 0,
      });
    const speech = () =>
      wllama.createSpeech({ input: 'Hello world', n_predict: 16, seed: 42 });
    const solo = { chat: await chat(), speech: await speech() };

    const session = await wllama.createRealtimeSession({ seed: 42 });
    await session.send({
      type: 'session.update',
      session: { output_modalities: ['audio', 'text'], max_output_tokens: 24 },
    });
    await session.appendAudio(createTestPcm(1));

    // the session is idle, waiting for audio
    expect((await chat()).choices[0].message.content).toBe(
      solo.chat.choices[0].message.content
    );

    // all of them at once, while the session generates a response
    await session.send({ type: 'input_audio_buffer.commit' });
    await session.send({ type: 'response.create' });
    const [events, parallelChat, parallelSpeech, transcript] =
      await Promise.all([
        readUntil(session, (e) => e.type === 'response.done'),
        chat(),
        speech(),
        wllama.createTranscription({ file: createTestWav(), max_tokens: 4 }),
      ]);
    expect(events.filter((e) => e.type === 'error')).toEqual([]);
    expect(events.at(-1)!.response.status).toBe('completed');
    expect(countEvents(events, 'response.output_audio.delta')).toBeGreaterThan(
      0
    );
    // the other slots must not change the results
    expect(parallelChat.choices[0].message.content).toBe(
      solo.chat.choices[0].message.content
    );
    expect(new Uint8Array(parallelSpeech.audio)).toEqual(
      new Uint8Array(solo.speech.audio)
    );
    expect(transcript.usage.output_tokens).toBe(4);

    // two sessions at once, and the first one still works
    const live = await wllama.createRealtimeSession({
      intent: 'transcription',
    });
    await live.appendAudio(createTestPcm(1));
    await live.send({ type: 'input_audio_buffer.commit', final: true });
    await session.send({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Hello' }],
      },
    });
    await session.send({
      type: 'response.create',
      response: { output_modalities: ['text'] },
    });
    const [liveEvents, next] = await Promise.all([
      readUntil(live, (e) =>
        e.type.endsWith('input_audio_transcription.completed')
      ),
      readUntil(session, (e) => e.type === 'response.done'),
    ]);
    expect(liveEvents.filter((e) => e.type === 'error')).toEqual([]);
    expect(next.filter((e) => e.type === 'error')).toEqual([]);
    expect(next.at(-1)!.response.status).toBe('completed');

    await live.close();
    await session.close();
    await wllama.exit();
  },
  120000
);

test.sequential(
  'keeps unread realtime events for the next reader',
  async () => {
    const wllama = await loadAudioModel();
    const session = await wllama.createRealtimeSession();
    // session.created and session.updated can come in the same poll
    await session.send({ type: 'session.update', session: {} });
    const first = await readUntil(session, (e) => e.type === 'session.created');
    expect(first.at(-1)!.type).toBe('session.created');
    const next = await readUntil(session, (e) => e.type === 'session.updated');
    expect(next.at(-1)!.type).toBe('session.updated');
    await session.close();
    await wllama.exit();
  },
  120000
);

test.sequential(
  'rejects a realtime session when no slot is free',
  async () => {
    // with one slot, a realtime start only works if nothing else holds it
    const wllama = createWllama();
    await wllama.loadModelFromUrl(AUDIO_MODEL, { n_ctx: 2048, n_parallel: 1 });
    const first = await wllama.createRealtimeSession();
    await expect(wllama.createRealtimeSession()).rejects.toThrow(
      'No free slot'
    );
    await first.close();

    // a running completion holds the slot
    const chat = wllama.createChatCompletion({
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 32,
    });
    await expect(wllama.createRealtimeSession()).rejects.toThrow(
      'No free slot'
    );
    expect((await chat).usage!.completion_tokens).toBeGreaterThan(0);

    // an aborted speech stream that is never read must free the slot
    const abort = new AbortController();
    await wllama.createSpeech({
      input: 'Hello world',
      stream: true,
      abortSignal: abort.signal,
    });
    abort.abort();
    const second = await wllama.createRealtimeSession();
    await second.close();
    await wllama.exit();
  },
  120000
);
