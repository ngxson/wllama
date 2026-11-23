// single-thread: only the part of the model on the CPU counts for the 4GB limit
import { test, expect } from 'vitest';
import { Wllama } from '../wllama';
import { RPC_CONFIG } from '../utils';

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

const load = async (nGpuLayers: number) => {
  const logs: string[] = [];
  const log = (...args: any[]) => logs.push(args.join(' '));
  const wllama = new Wllama(
    { default: '/src/wasm/wllama.wasm' },
    { logger: { debug: log, log, warn: log, error: log } }
  );
  wllama.setCompat(null);
  try {
    await wllama.loadModelFromUrl(TINY_MODEL, {
      n_ctx: 512,
      n_threads: 1,
      n_gpu_layers: nGpuLayers,
    });
    return logs;
  } finally {
    await wllama.exit();
  }
};

test('single-thread loads a model over the limit when its weights are on WebGPU', async () => {
  const config = { ...RPC_CONFIG };
  // the model has about 17MiB of weights
  Object.assign(RPC_CONFIG, { autoMinSize: 0, singleThreadMemMax: 10 });
  try {
    await expect(load(0)).rejects.toThrow(
      'single-thread build does not support'
    );
    const logs = await load(99);
    console.log(logs.filter((l) => /buffer size/.test(l)).join('\n'));
    expect(logs.some((l) => /WebGPU.*model buffer size/.test(l))).toBe(true);
  } finally {
    Object.assign(RPC_CONFIG, config);
  }
}, 60000);
