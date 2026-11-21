// a model in the OPFS cache is read with a sync access handle, and with its Blob when that fails
import { test, expect } from 'vitest';
import { Wllama } from '../wllama';
import { ModelManager } from '../model-manager';

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

const complete = async (load: (w: Wllama) => Promise<void>) => {
  const logs: string[] = [];
  const log = (...args: any[]) => logs.push(args.join(' '));
  const wllama = new Wllama(
    { default: '/src/wasm/wllama.wasm' },
    { logger: { debug: log, log, warn: log, error: log } }
  );
  wllama.setCompat(null);
  try {
    await load(wllama);
    const res = await wllama.createCompletion({
      prompt: 'Once upon a time,',
      max_tokens: 8,
      temperature: 0,
    });
    return { text: res.choices[0].text, logs };
  } finally {
    await wllama.exit();
  }
};

// the same model, not from the cache
const expected = async () => {
  const blob = await (await fetch(TINY_MODEL)).blob();
  return (await complete((w) => w.loadModel([blob], { n_threads: 2 }))).text;
};

test('reads a cached model with an OPFS sync access handle', async () => {
  const want = await expected();
  const { text, logs } = await complete((w) =>
    w.loadModelFromUrl(TINY_MODEL, { n_threads: 2 })
  );
  expect(logs.some((l) => l.includes('with an OPFS sync access handle'))).toBe(
    true
  );
  expect(logs.some((l) => l.includes('Cannot read'))).toBe(false);
  expect(text).toBe(want);
});

test('reads the Blob when the file is locked', async () => {
  const want = await expected();
  await new ModelManager().downloadModel(TINY_MODEL);
  // another worker holds an exclusive sync access handle on the cached file
  const key = await new ModelManager().cacheManager.getNameFromURL(TINY_MODEL);
  const locker = new Worker(
    URL.createObjectURL(
      new Blob(
        [
          // opened in the worker: a handle cannot be sent to a worker in Safari
          `onmessage = async (e) => {
            const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('cache');
            await (await dir.getFileHandle(e.data)).createSyncAccessHandle();
            postMessage('locked');
          };`,
        ],
        { type: 'text/javascript' }
      )
    )
  );
  await new Promise((resolve) => {
    locker.onmessage = resolve;
    locker.postMessage(key);
  });
  try {
    const { text, logs } = await complete((w) =>
      w.loadModelFromUrl(TINY_MODEL, { n_threads: 2 })
    );
    expect(logs.some((l) => l.includes('Cannot read'))).toBe(true);
    expect(text).toBe(want);
  } finally {
    locker.terminate();
  }
});
