import { test, expect, beforeEach } from 'vitest';
import { CacheManager } from '../cache-manager';
import { COSBackend, mockCOS } from './cos';
import type { StorageBackend } from './index';

let onCOSWrite: (() => void) | undefined;
let cosWriteError: Error | undefined;

async function randomBufAndHash(): Promise<{
  buf: Uint8Array;
  sha256: string;
}> {
  const buf = crypto.getRandomValues(new Uint8Array(256));
  const hashBuf = await crypto.subtle.digest('SHA-256', buf);
  const sha256 = Array.from(new Uint8Array(hashBuf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  return { buf, sha256 };
}

function bufStream(buf: Uint8Array): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(buf.slice());
      controller.close();
    },
  });
}

test.sequential('write then read without hint falls back to OPFS', async () => {
  const backend = new COSBackend();
  const { buf } = await randomBufAndHash();
  const key = 'test-no-hint';

  await backend.write(key, bufStream(buf));
  const blob = await backend.read(key);
  expect(blob).not.toBeNull();
  expect(new Uint8Array(await blob!.arrayBuffer())).toEqual(buf);

  await backend.delete(key);
});

beforeEach(() => {
  onCOSWrite = undefined;
  cosWriteError = undefined;
  mockCOS({
    onWrite: () => onCOSWrite?.(),
    writeError: () => cosWriteError,
  });
});

test.sequential('write with hint stores in COS only', async () => {
  const backend = new COSBackend();
  expect(backend.isSupported()).toBe(true);

  const { buf, sha256 } = await randomBufAndHash();
  const hint = { sha256 };
  const key = 'test-with-hint';

  await backend.write(key, bufStream(buf), hint);

  // read back via hint → should hit COS
  const blob = await backend.read(key, hint);
  expect(blob).not.toBeNull();
  expect(new Uint8Array(await blob!.arrayBuffer())).toEqual(buf);

  // without hint → OPFS fallback → not found (was written to COS only)
  const blobOpfs = await backend.read(key);
  expect(blobOpfs).toBeNull();

  await backend.delete(key);
});

test.sequential('getSize with hint reflects COS size', async () => {
  const backend = new COSBackend();
  const { buf, sha256 } = await randomBufAndHash();
  const hint = { sha256 };
  const key = 'test-size-hint';

  await backend.write(key, bufStream(buf), hint);

  const size = await backend.getSize(key, hint);
  expect(size).toBe(buf.byteLength);

  await backend.delete(key);
});

test.sequential(
  'aborted COS writes are discarded and recoverable',
  async () => {
    const backend = new COSBackend();
    const cache = new CacheManager([backend]);
    const { buf, sha256 } = await randomBufAndHash();
    const hint = { sha256 };
    const url = `https://huggingface.co/example/model/resolve/main/model.gguf?${crypto.randomUUID()}`;
    const originalFetch = globalThis.fetch;
    let partialWasWritten = false;
    onCOSWrite = () => {
      partialWasWritten = true;
    };
    cosWriteError = new DOMException('The operation was aborted', 'AbortError');
    const writing = backend.write('unused', bufStream(buf.slice(0, 64)), hint);
    await expect(writing).rejects.toThrow('aborted');

    expect(partialWasWritten).toBe(true);
    expect(await backend.read('unused', hint)).toBeNull();

    // A partial object left by an older writer must not satisfy the cache hit.
    cosWriteError = undefined;
    await backend.write('unused', bufStream(buf.slice(0, 64)), hint);
    expect(await backend.getSize('unused', hint)).toBe(64);

    let downloadCount = 0;
    globalThis.fetch = ((input, init) => {
      const requestUrl = String(input);
      if (requestUrl.includes('/raw/')) {
        return Promise.resolve(new Response(`oid sha256:${sha256}`));
      }
      if (init?.method === 'HEAD') {
        return Promise.resolve(
          new Response(null, {
            headers: { 'content-length': String(buf.byteLength) },
          })
        );
      }

      downloadCount++;
      return Promise.resolve(
        new Response(buf.slice(), {
          headers: { 'content-length': String(buf.byteLength) },
        })
      );
    }) as typeof fetch;

    try {
      await cache.download(url);
      const blob = await backend.read('unused', hint);
      expect(new Uint8Array(await blob!.arrayBuffer())).toEqual(buf);
      expect(downloadCount).toBe(1);
    } finally {
      globalThis.fetch = originalFetch;
      await cache.delete(url);
    }
  }
);

test.sequential('failed COS writes cancel their source stream', async () => {
  const backend = new COSBackend();
  const { buf, sha256 } = await randomBufAndHash();
  const writeError = new Error('COS write failed');
  let cancelReason: unknown;

  cosWriteError = writeError;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(buf);
    },
    cancel(reason) {
      cancelReason = reason;
      throw new Error('Source cancellation failed');
    },
  });

  await expect(backend.write('unused', stream, { sha256 })).rejects.toBe(
    writeError
  );
  expect(cancelReason).toBe(writeError);
  expect(stream.locked).toBe(false);
});

test.sequential(
  'cache listing rediscovers COS data from metadata',
  async () => {
    const backend = new COSBackend();
    const cache = new CacheManager([backend]);
    const { buf, sha256 } = await randomBufAndHash();
    const key = 'test-cached-model';

    await backend.write(key, bufStream(buf), { sha256 });
    await cache.writeMetadata(key, {
      etag: 'fixture-etag',
      originalSize: buf.byteLength,
      originalURL: 'https://example.com/model.gguf',
      sha256,
    });

    expect(await cache.list()).toContainEqual({
      metadata: {
        etag: 'fixture-etag',
        originalSize: buf.byteLength,
        originalURL: 'https://example.com/model.gguf',
        sha256,
      },
      name: key,
      size: buf.byteLength,
    });
    expect(
      new Uint8Array(await (await cache.open(key))!.arrayBuffer())
    ).toEqual(buf);
  }
);

test.sequential('read missing key returns null', async () => {
  const backend = new COSBackend();
  const { sha256 } = await randomBufAndHash();
  const blob = await backend.read('non-existent-key', { sha256 });
  expect(blob).toBeNull();
});

test.sequential(
  'abort signal cancels cached model metadata lookup',
  async () => {
    const sha256 = 'a'.repeat(64);
    const writes: string[] = [];
    const backend: StorageBackend = {
      isSupported: () => true,
      read: async () => null,
      write: async (key) => {
        writes.push(key);
      },
      getSize: async (_key, hint) => (hint ? 1024 : -1),
      list: async () => [],
      delete: async () => {},
    };
    const cache = new CacheManager([backend]);
    const originalFetch = globalThis.fetch;
    let headSignal!: AbortSignal;
    let markHeadStarted!: () => void;
    const headStarted = new Promise<void>((resolve) => {
      markHeadStarted = resolve;
    });
    globalThis.fetch = ((input, init) => {
      const url = String(input);
      if (url.includes('/raw/')) {
        return Promise.resolve(new Response(`oid sha256:${sha256}`));
      }
      if (init?.method === 'HEAD') {
        headSignal = init.signal!;
        markHeadStarted();
        return new Promise((_, reject) => {
          headSignal.addEventListener(
            'abort',
            () =>
              reject(
                new DOMException('The operation was aborted', 'AbortError')
              ),
            { once: true }
          );
        });
      }
      throw new Error(`Unexpected request: ${url}`);
    }) as typeof fetch;

    const controller = new AbortController();
    const downloading = cache.download(
      'https://huggingface.co/example/model/resolve/main/model.gguf',
      { signal: controller.signal }
    );
    try {
      await headStarted;
      const rejected = expect(downloading).rejects.toThrow('aborted');
      controller.abort();

      expect(headSignal.aborted).toBe(true);
      await rejected;
      expect(writes).toEqual([]);
    } finally {
      controller.abort();
      globalThis.fetch = originalFetch;
    }
  }
);
