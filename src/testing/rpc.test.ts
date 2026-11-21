// the model is split across wasm instances in other workers
import { test, expect } from 'vitest';
import { Wllama } from '../wllama';
import { RPC_CONFIG } from '../utils';
import { makeFakeModel } from './fake-gguf';

declare const __COMPAT__: boolean;

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

const run = async (params: Record<string, any>) => {
  const logs: string[] = [];
  const log = (...args: any[]) => logs.push(args.join(' '));
  const wllama = new Wllama(
    { default: '/src/wasm/wllama.wasm' },
    { logger: { debug: log, log, warn: log, error: log } }
  );
  wllama.setCompat(
    __COMPAT__ ? { wasm: '/compat/wasm/wllama.wasm' } : null,
    'firefox_safari'
  );
  try {
    await wllama.loadModelFromUrl(TINY_MODEL, { n_ctx: 512, ...params });
    const res = await wllama.createCompletion({
      prompt: 'Once upon a time, there was a little girl named Lily.',
      max_tokens: 16,
      temperature: 0,
    });
    return { text: res.choices[0].text, logs };
  } catch (e) {
    // the native logs are only in `logs`, show them to see why it failed
    console.log(logs.slice(-30).join('\n'));
    throw e;
  } finally {
    await wllama.exit();
  }
};

test('splits the model across rpc workers', async () => {
  const cpu = await run({ n_gpu_layers: 0, n_threads: 2 });
  const rpc = await run({ rpc_workers: 2, n_gpu_layers: 99, n_threads: 2 });
  console.log(
    'BUFFERS',
    rpc.logs.filter((l) => l.includes('model buffer size')).join(' | ')
  );
  console.log('CPU', JSON.stringify(cpu.text), 'RPC', JSON.stringify(rpc.text));
  expect(rpc.logs.some((l) => /RPC\d+\[.*\] model buffer size/.test(l))).toBe(
    true
  );
  expect(rpc.text).toBe(cpu.text);
}, 60000);

test('uses rpc workers when the model does not fit in the main instance', async () => {
  const cpu = await run({ n_gpu_layers: 0, n_threads: 2 });
  // measure all models, and allow 1MiB in the main instance
  const config = { ...RPC_CONFIG };
  Object.assign(RPC_CONFIG, { autoMinSize: 0, mainMemMax: 1 });
  try {
    const rpc = await run({ n_gpu_layers: 0, n_threads: 2 });
    expect(rpc.logs.some((l) => /using 1 rpc workers/.test(l))).toBe(true);
    expect(rpc.logs.some((l) => /RPC0\[.*\] model buffer size/.test(l))).toBe(
      true
    );
    expect(rpc.text).toBe(cpu.text);
  } finally {
    Object.assign(RPC_CONFIG, config);
  }
}, 60000);

test('rpc workers need async file reads', async () => {
  // without JSPI and without compat, the model is copied into the main wasm, so rpc workers cannot help
  const suspending = (WebAssembly as any).Suspending;
  delete (WebAssembly as any).Suspending;
  const wllama = new Wllama({ default: '/src/wasm/wllama.wasm' });
  wllama.setCompat(null);
  const model = makeFakeModel({
    nEmbd: 64,
    nFf: 128,
    nLayer: 2,
    nHead: 4,
    shardSize: 1024 * 1024 * 1024,
  });
  try {
    await expect(
      wllama.loadModel(model, { rpc_workers: 2, n_threads: 2 })
    ).rejects.toThrow('rpc workers need a browser with JSPI');
    expect(wllama.isModelLoaded()).toBe(false);
  } finally {
    if (suspending) (WebAssembly as any).Suspending = suspending;
    await wllama.exit();
  }
});
