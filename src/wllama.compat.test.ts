import { expect, test } from 'vitest';
import COMPAT_EMSCRIPTEN_CODE from '../compat/wasm/wllama.js?raw';
import { Wllama } from './wllama';

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

test.sequential(
  'the wasm32 compat artifact runs real inference',
  async () => {
    const compatWasmResponse = await fetch('/compat/wasm/wllama.wasm');
    const compatWasmMagic = new Uint8Array(
      await compatWasmResponse.clone().arrayBuffer(),
      0,
      4
    );

    expect(compatWasmResponse.ok).toBe(true);
    expect([...compatWasmMagic]).toEqual([0x00, 0x61, 0x73, 0x6d]);

    const nativeSuspendingDescriptor = Object.getOwnPropertyDescriptor(
      WebAssembly,
      'Suspending'
    );
    let rejectWorkerError!: (reason: Error) => void;
    const workerError = new Promise<never>((_, reject) => {
      rejectWorkerError = reject;
    });
    const wllama = new Wllama(
      { default: '/src/wasm/wllama.wasm' },
      {
        logger: {
          debug: console.debug,
          log: console.log,
          warn: console.warn,
          error: (error: unknown, ...details: unknown[]) => {
            console.error(error, ...details);
            if (error instanceof Event) {
              const workerEvent = error as ErrorEvent;
              rejectWorkerError(
                new Error(
                  [
                    workerEvent.message || 'Wllama worker failed',
                    workerEvent.filename,
                    workerEvent.lineno,
                    workerEvent.colno,
                  ]
                    .filter(Boolean)
                    .join(':')
                )
              );
            }
          },
        },
      }
    );

    // A browser without JSPI normally selects compat. Hide that capability only
    // in the main page; the isolated worker still executes the real wasm32
    // Asyncify runtime with its native WebAssembly implementation.
    Object.defineProperty(WebAssembly, 'Suspending', {
      configurable: true,
      value: undefined,
    });

    try {
      wllama.setCompat({
        wasm: '/compat/wasm/wllama.wasm',
        worker: { code: COMPAT_EMSCRIPTEN_CODE },
      });

      await Promise.race([
        wllama.loadModelFromUrl(TINY_MODEL, {
          n_ctx: 256,
          n_gpu_layers: 0,
          n_threads: 1,
        }),
        workerError,
      ]);

      const completion = await wllama.createCompletion({
        prompt: 'Once upon a time',
        max_tokens: 4,
        seed: 42,
        temperature: 0,
      });

      expect(wllama.getWorkerResources().compat).toBe(true);
      expect(wllama.isModelLoaded()).toBe(true);
      expect(completion.choices[0].text.length).toBeGreaterThan(0);
    } finally {
      try {
        await wllama.exit();
      } finally {
        if (nativeSuspendingDescriptor) {
          Object.defineProperty(
            WebAssembly,
            'Suspending',
            nativeSuspendingDescriptor
          );
        } else {
          delete (WebAssembly as any).Suspending;
        }
      }
    }
  },
  120_000
);
