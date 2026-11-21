import { test, expect, afterEach } from 'vitest';
import { GGUF_SPLIT_CONFIG, splitBigGguf } from './gguf-split';
import { Wllama } from './wllama';

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

const DEFAULT_CONFIG = { ...GGUF_SPLIT_CONFIG };
afterEach(() => Object.assign(GGUF_SPLIT_CONFIG, DEFAULT_CONFIG));

// split the tiny model as if it was bigger than the limit
const forceSplit = () =>
  Object.assign(GGUF_SPLIT_CONFIG, {
    maxFileSize: 1024 * 1024,
    shardSize: 4 * 1024 * 1024,
  });

// downloaded once for all tests
let tinyModel: Promise<Blob> | null = null;
const getTinyModel = async () => {
  tinyModel ??= fetch(TINY_MODEL).then((r) => r.blob());
  return new File([await tinyModel], 'stories15M-q4_0.gguf');
};

const readU64 = (b: Uint8Array, pos: number) =>
  Number(new DataView(b.buffer, b.byteOffset).getBigUint64(pos, true));

test('keeps files under the limit as they are', async () => {
  const file = await getTinyModel();
  expect(await splitBigGguf(file)).toEqual([file]);
});

test('splits a big file into virtual shards', async () => {
  const file = await getTinyModel();
  forceSplit();
  const shards = await splitBigGguf(file);
  expect(shards.length).toBeGreaterThan(2);
  shards.forEach((s, i) =>
    expect((s as File).name).toBe(
      `stories15M-q4_0-0000${i + 1}-of-0000${shards.length}.gguf`
    )
  );

  // the tensor counts of the shards add up, and the last shard ends with the end of the file
  const orig = new Uint8Array(await file.arrayBuffer());
  let total = 0;
  for (const s of shards) {
    total += readU64(new Uint8Array(await s.arrayBuffer()), 8);
  }
  expect(total).toBe(readU64(orig, 8));
  const last = new Uint8Array(await shards[shards.length - 1].arrayBuffer());
  expect(last.subarray(last.byteLength - 1024)).toEqual(
    orig.subarray(orig.byteLength - 1024)
  );
});

test('loads a model split into virtual shards', async () => {
  const run = async (split: boolean) => {
    const logs: string[] = [];
    const log = (...args: any[]) => logs.push(args.join(' '));
    if (split) forceSplit();
    const wllama = new Wllama(
      { default: '/src/wasm/wllama.wasm' },
      { logger: { debug: log, log, warn: log, error: log } }
    );
    wllama.setCompat(null);
    await wllama.loadModel([await getTinyModel()], {
      n_ctx: 512,
      n_threads: 1,
      n_gpu_layers: 0,
    });
    const res = await wllama.createCompletion({
      prompt: 'Once upon a time, there was a little girl named Lily.',
      max_tokens: 16,
      temperature: 0,
    });
    await wllama.exit();
    return { text: res.choices[0].text, logs };
  };
  const plain = await run(false);
  const split = await run(true);
  expect(
    split.logs.some((l) => /additional \d+ GGUFs metadata loaded/.test(l))
  ).toBe(true);
  expect(split.text).toBe(plain.text);
});

test('a failed split does not break the next load', async () => {
  // 4GB that is not a GGUF, slices are zeros
  const size = 4 * 1024 * 1024 * 1024;
  const bad = {
    size,
    name: 'bad.gguf',
    slice: (a = 0, b = size) =>
      new Blob([new Uint8Array(Math.min(b - a, 1024 * 1024))]),
  } as unknown as Blob;
  const wllama = new Wllama({ default: '/src/wasm/wllama.wasm' });
  wllama.setCompat(null);
  await expect(wllama.loadModel([bad])).rejects.toThrow('Not a GGUF file');
  await wllama.loadModel([await getTinyModel()], {
    n_ctx: 512,
    n_threads: 1,
    n_gpu_layers: 0,
  });
  expect(wllama.isModelLoaded()).toBe(true);
  await wllama.exit();
});
