import { test, expect, afterEach } from 'vitest';
import { GGUF_SPLIT_CONFIG, splitBigGguf } from '../gguf-split';
import { Wllama } from '../wllama';
import { classifyGguf, prepareBlobs } from '../utils';
import { makeGgufHeader } from './fake-gguf';

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

test('tells the model, mmproj, draft and LoRA files apart', async () => {
  const layer = (i: number) => [
    `blk.${i}.attn_q.weight`,
    `blk.${i}.ffn_down.weight`,
  ];
  const nextn = (i: number) => [...layer(i), `blk.${i}.nextn.eh_proj.weight`];
  const role = (p: Parameters<typeof makeGgufHeader>[0]) =>
    classifyGguf(makeGgufHeader(p));

  expect(
    await role({
      arch: 'llama',
      tensors: ['token_embd.weight', ...layer(0), ...layer(1)],
    })
  ).toEqual({ role: 'model' });
  expect(
    await role({ arch: 'clip', tensors: ['v.patch_embd.weight'] })
  ).toEqual({ role: 'mmproj' });
  expect(
    await role({
      arch: 'llama',
      type: 'adapter',
      tensors: ['blk.0.attn_q.weight.lora_a'],
    })
  ).toEqual({ role: 'lora' });
  expect(await role({ arch: 'dflash', tensors: layer(0) })).toEqual({
    role: 'draft',
    spec_type: 'draft-dflash',
  });
  expect(await role({ arch: 'eagle3', tensors: layer(0) })).toEqual({
    role: 'draft',
    spec_type: 'draft-eagle3',
  });
  // an MTP sidecar only has the blocks of the MTP layers
  expect(
    await role({ arch: 'qwen35', tensors: ['token_embd.weight', ...nextn(32)] })
  ).toEqual({ role: 'draft', spec_type: 'draft-mtp' });
  // a model with its MTP layers inside, and the last shard of a split model, are the model
  expect(
    await role({
      arch: 'qwen35',
      tensors: [...layer(0), ...layer(1), ...nextn(2)],
    })
  ).toEqual({ role: 'model' });
  expect(
    await role({ arch: 'qwen35', splitCount: 3, tensors: nextn(32) })
  ).toEqual({ role: 'model' });
});

test('prepares the model, mmproj, draft and LoRA files', async () => {
  const model = makeGgufHeader({
    arch: 'llama',
    tensors: ['blk.0.attn_q.weight'],
  });
  const mmproj = makeGgufHeader({
    arch: 'clip',
    tensors: ['v.patch_embd.weight'],
  });
  const draft = makeGgufHeader({
    arch: 'dflash',
    tensors: ['blk.0.attn_q.weight'],
  });
  const lora = () =>
    makeGgufHeader({
      arch: 'llama',
      type: 'adapter',
      tensors: ['blk.0.attn_q.weight.lora_a'],
    });

  const files = await prepareBlobs([lora(), draft, model, mmproj, lora()]);
  expect(files.llm.map((f) => f.name)).toEqual(['model-00001-of-00001.gguf']);
  expect(files.mmproj?.name).toBe('mmproj.gguf');
  expect(files.draft).toMatchObject({
    name: 'draft.gguf',
    spec_type: 'draft-dflash',
  });
  expect(files.loras.map((f) => f.name)).toEqual([
    'lora-1.gguf',
    'lora-2.gguf',
  ]);
  expect(files.all.map((f) => f.name)).toEqual([
    'model-00001-of-00001.gguf',
    'mmproj.gguf',
    'draft.gguf',
    'lora-1.gguf',
    'lora-2.gguf',
  ]);

  // a normal model given as the draft runs with draft-simple
  const small = makeGgufHeader({
    arch: 'llama',
    tensors: ['blk.0.attn_q.weight'],
  });
  expect((await prepareBlobs([model], small)).draft?.spec_type).toBe(
    'draft-simple'
  );

  await expect(prepareBlobs([model, draft], draft)).rejects.toThrow(
    'More than one speculative draft'
  );
  await expect(prepareBlobs([mmproj])).rejects.toThrow('No model file');
});
