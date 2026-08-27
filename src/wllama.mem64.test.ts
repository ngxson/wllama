import { test, expect } from 'vitest';
import { Wllama } from './wllama';

// POC for Memory64: alloc 14GB inside wasm via heapfs, way past the old 4GB cap
// run with: npm run test:mem64
// warning: needs ~16GB of free RAM

const SIZE_GB = 14;
const GB = 1024 * 1024 * 1024;

const CONFIG_PATHS = {
  default: '/src/wasm/wllama.wasm',
};

test.sequential(
  'allocates more than 4GB with mem64',
  async () => {
    const wllama = new Wllama(CONFIG_PATHS);
    wllama.setCompat(null);

    const res = await wllama._testMem64Alloc(SIZE_GB);
    console.log('mem64 alloc result:', res);

    expect(res.allocated_bytes).toBe(SIZE_GB * GB);
    expect(res.write_offsets_ok).toBe(4);
  },
  600_000
);

// smoke test with a real model bigger than 4GB, only runs if the file is present locally
// to set it up, hardlink (or copy) a 4GB+ gguf into ./models
//
// note: we cannot use loadModelFromUrl here because chromium caps OPFS at ~2GB in the
// ephemeral (incognito) context used by vitest. Instead, we bypass the cache with a
// lazy Blob that serves slice() reads via HTTP range requests.
const BIG_MODEL_URL = '/models/Qwen3.5-4B-BF16.gguf';
const bigModelAvailable = await fetch(BIG_MODEL_URL, { method: 'HEAD' })
  .then((res) => res.ok)
  .catch(() => false);

class UrlBlob extends Blob {
  constructor(
    private url: string,
    private _size: number,
    private offset: number = 0
  ) {
    super();
  }
  get size(): number {
    return this._size;
  }
  slice(start: number = 0, end: number = this._size): Blob {
    return new UrlBlob(this.url, end - start, this.offset + start);
  }
  async arrayBuffer(): Promise<ArrayBuffer> {
    const res = await fetch(this.url, {
      headers: {
        Range: `bytes=${this.offset}-${this.offset + this._size - 1}`,
      },
    });
    if (!res.ok) {
      throw new Error(`range request failed: HTTP ${res.status}`);
    }
    return res.arrayBuffer();
  }
}

test.skipIf(!bigModelAvailable).sequential(
  'loads a model bigger than 4GB',
  async () => {
    const res0 = await fetch(BIG_MODEL_URL, {
      headers: { Range: 'bytes=0-0' },
    });
    const totalSize = parseInt(
      (res0.headers.get('content-range') ?? '/0').split('/')[1],
      10
    );
    console.log('model file size:', totalSize);
    expect(totalSize).toBeGreaterThan(4 * GB);

    const wllama = new Wllama(CONFIG_PATHS);
    wllama.setCompat(null);

    // n_threads: 1 because chromium caps shared memory64 growth at 4GB when 2 or more
    // pthread workers exist (fine with 0 or 1). Needs an upstream report to V8/emscripten.
    await wllama.loadModel([new UrlBlob(BIG_MODEL_URL, totalSize)], {
      n_ctx: 512,
      n_threads: 1,
    });
    expect(wllama.isModelLoaded()).toBe(true);

    const res = await wllama.createCompletion({
      prompt: 'The capital of France is',
      max_tokens: 8,
      temperature: 0.0,
    });
    console.log('smoke completion:', res.choices[0].text);
    expect(res.choices[0].text.length).toBeGreaterThan(0);

    await wllama.exit();
  },
  1800_000
);
