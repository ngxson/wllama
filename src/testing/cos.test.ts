import { test, expect, beforeEach } from 'vitest';
import { COSBackend, mockCOS } from '../storage/cos';
import { OPFSBackend } from '../storage/opfs';
import CacheManager from '../cache-manager';
import { ModelManager } from '../model-manager';

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf';

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
  mockCOS();
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

test.sequential('read missing key returns null', async () => {
  const backend = new COSBackend();
  const { sha256 } = await randomBufAndHash();
  const blob = await backend.read('non-existent-key', { sha256 });
  expect(blob).toBeNull();
});

test.sequential(
  'write falls back to OPFS when COS is not allowed',
  async () => {
    let calls = 0;
    (navigator as any).crossOriginStorage = {
      getFileHandle: () => {
        calls++;
        return Promise.reject(
          new DOMException('Not allowed', 'NotAllowedError')
        );
      },
    };
    const backend = new COSBackend();
    const { buf, sha256 } = await randomBufAndHash();
    const key = 'test-not-allowed';

    await backend.write(key, bufStream(buf), { sha256 });
    const blob = await backend.read(key);
    expect(new Uint8Array(await blob!.arrayBuffer())).toEqual(buf);

    // the next write skips COS
    await backend.write(key, bufStream(buf), { sha256 });
    expect(calls).toBe(1);

    await backend.delete(key);
  }
);

test.sequential(
  'downloaded model stored in COS is found in the cache',
  async () => {
    const manager = new ModelManager({
      cacheManager: new CacheManager([new COSBackend()]),
    });
    await manager.clear();
    const model = await manager.downloadModel(TINY_MODEL);
    expect(model.validate()).toBe('valid');

    // the file is in COS, OPFS has only its metadata
    const fileKey = await manager.cacheManager.getNameFromURL(TINY_MODEL);
    expect(await new OPFSBackend().getSize(fileKey)).toBe(-1);

    const models = await manager.getModels();
    expect(models.find((m) => m.url === TINY_MODEL)?.validate()).toBe('valid');
    const [blob] = await model.open();
    expect(blob.size).toBe(model.size);

    await manager.clear();
  }
);

test.sequential(
  'COS blocked by Permissions Policy falls back to OPFS',
  async () => {
    (navigator as any).crossOriginStorage = {
      getFileHandle: () =>
        Promise.reject(
          new DOMException(
            'Cross-Origin Storage is disabled by Permissions Policy in this context.',
            'NotAllowedError'
          )
        ),
    };
    const manager = new ModelManager({
      cacheManager: new CacheManager([new COSBackend()]),
    });
    await manager.clear();
    const model = await manager.downloadModel(TINY_MODEL);
    expect(model.validate()).toBe('valid');

    const fileKey = await manager.cacheManager.getNameFromURL(TINY_MODEL);
    expect(await new OPFSBackend().getSize(fileKey)).toBe(model.size);

    await manager.clear();
  }
);
