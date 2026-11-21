// A 6GB model, made on the fly (see fake-gguf.ts). Opt-in only: BIG_MODEL=1, it needs WebGPU and about 12GB of free RAM.
import { test, expect } from 'vitest';
import { Wllama } from '../wllama';
import { makeFakeModel } from './fake-gguf';

// like a 7B: 28 layers of about 215MB (Q8_0), about 6.0GB in total
const N_LAYER = 28;
const model = () =>
  makeFakeModel({
    nEmbd: 4096,
    nFf: 11008,
    nLayer: N_LAYER,
    nHead: 32,
    shardSize: 2 * 1024 * 1024 * 1024,
  });

const run = async (n_gpu_layers: number) => {
  const logs: string[] = [];
  const log = (...args: any[]) => logs.push(args.join(' '));
  const wllama = new Wllama(
    { default: '/src/wasm/wllama.wasm' },
    { logger: { debug: log, log, warn: log, error: log } }
  );
  wllama.setCompat(null);
  try {
    await wllama.loadModel(model(), { n_ctx: 256, n_threads: 4, n_gpu_layers });
    const res = await wllama.createCompletion({
      prompt: 'Hello',
      max_tokens: 4,
      temperature: 0,
      ignore_eos: true,
    } as any);
    expect(res.timings?.predicted_n).toBe(4);
    return logs.filter((l) => /model buffer size/.test(l));
  } catch (e) {
    console.log(logs.slice(-20).join('\n'));
    throw e;
  } finally {
    await wllama.exit();
  }
};

test('6GB model on CPU only (rpc workers)', async () => {
  const buffers = await run(0);
  expect(buffers.some((l) => /RPC\d+\[.*\] model buffer size/.test(l))).toBe(
    true
  );
}, 600000);

test('6GB model on GPU only', async () => {
  const buffers = await run(99999);
  expect(buffers.some((l) => /WebGPU model buffer size/.test(l))).toBe(true);
}, 600000);

// about 5GB of layers on the CPU: they stay in the main wasm (4GB), rpc workers are not used with WebGPU
test.fails(
  '6GB model, 5GB on CPU and the rest on GPU',
  async () => {
    await run(5);
  },
  600000
);
