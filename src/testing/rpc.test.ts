// the model is split across wasm instances in other workers
import { test, expect } from 'vitest';
import { Wllama } from '../wllama';

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
