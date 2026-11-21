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
  // without JSPI (Safari, Firefox, COMPAT=1), the compat build reads the files the same way
  wllama.setCompat({ wasm: '/compat/wasm/wllama.wasm' }, 'firefox_safari');
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

// another worker opens an exclusive sync access handle on the cached file (it fails while someone else has it open)
// it is opened in the worker: a handle cannot be sent to a worker in Safari
const openInWorker = async (url: string) => {
  const key = await new ModelManager().cacheManager.getNameFromURL(url);
  const worker = new Worker(
    URL.createObjectURL(
      new Blob(
        [
          `let access;
          onmessage = async (e) => {
            try {
              if (e.data === 'close') { access.close(); postMessage('closed'); return; }
              const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('cache');
              access = await (await dir.getFileHandle(e.data)).createSyncAccessHandle();
              postMessage('opened');
            } catch (err) { postMessage('failed: ' + err); }
          };`,
        ],
        { type: 'text/javascript' }
      )
    )
  );
  const call = (msg: string) =>
    new Promise<string>((resolve) => {
      worker.onmessage = (e) => resolve(e.data);
      worker.postMessage(msg);
    });
  return { result: await call(key), close: () => call('close'), worker };
};

// the same model, not from the cache
const expected = async () => {
  const blob = await (await fetch(TINY_MODEL)).blob();
  return (await complete((w) => w.loadModel([blob], { n_threads: 2 }))).text;
};

test('reads a cached model with an OPFS sync access handle', async () => {
  const want = await expected();
  const { text, logs } = await complete(async (w) => {
    await w.loadModelFromUrl(TINY_MODEL, { n_threads: 2 });
    // the model stays loaded, but its handle is closed after each wasm call: the cache can open the file
    const other = await openInWorker(TINY_MODEL);
    expect(other.result).toBe('opened');
    await other.close();
    other.worker.terminate();
  });
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
  const locker = await openInWorker(TINY_MODEL);
  expect(locker.result).toBe('opened');
  try {
    const { text, logs } = await complete((w) =>
      w.loadModelFromUrl(TINY_MODEL, { n_threads: 2 })
    );
    expect(logs.some((l) => l.includes('Cannot read'))).toBe(true);
    expect(text).toBe(want);
  } finally {
    locker.worker.terminate();
  }
});
