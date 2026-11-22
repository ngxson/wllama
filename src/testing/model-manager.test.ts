import { test, expect, onTestFinished } from 'vitest';
import { ModelManager, Model, ModelValidationStatus } from '../model-manager';
import { OPFSBackend } from '../storage/opfs';
import CacheManager, { DOWNLOAD_CONFIG } from '../cache-manager';
import { hookNetwork, type NetworkHook } from './network-hook';
import { getHFFileSHA256 } from '../huggingface';

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf';
const SPLIT_MODEL =
  'https://huggingface.co/ngxson/tinyllama_split_test/resolve/main/stories15M-q8_0-00001-of-00003.gguf';

test.sequential('parseModelUrl handles single model URL', () => {
  const urls = ModelManager.parseModelUrl(TINY_MODEL);
  expect(urls.length).toBe(1);
  expect(urls[0]).toBe(TINY_MODEL);
});

test.sequential('parseModelUrl handles array of URLs', () => {
  const urls = ModelManager.parseModelUrl(SPLIT_MODEL);
  expect(urls.length).toBe(3);
  expect(urls[0]).toMatch(/-00001-of-00003\.gguf$/);
  expect(urls[1]).toMatch(/-00002-of-00003\.gguf$/);
  expect(urls[2]).toMatch(/-00003-of-00003\.gguf$/);
});

test.sequential('parseModelUrl handles URLs with query parameters', () => {
  // Test with a simple query parameter
  const urlWithQuery =
    'https://example.com/models/model-00001-of-00003.gguf?param=value';
  const urls = ModelManager.parseModelUrl(urlWithQuery);
  expect(urls.length).toBe(3);
  expect(urls[0]).toBe(
    'https://example.com/models/model-00001-of-00003.gguf?param=value'
  );
  expect(urls[1]).toBe(
    'https://example.com/models/model-00002-of-00003.gguf?param=value'
  );
  expect(urls[2]).toBe(
    'https://example.com/models/model-00003-of-00003.gguf?param=value'
  );

  // Test with multiple query parameters
  const urlWithMultipleParams =
    'https://example.com/models/model-00001-of-00002.gguf?param1=value1&param2=value2';
  const urlsMultiParams = ModelManager.parseModelUrl(urlWithMultipleParams);
  expect(urlsMultiParams.length).toBe(2);
  expect(urlsMultiParams[0]).toBe(
    'https://example.com/models/model-00001-of-00002.gguf?param1=value1&param2=value2'
  );
  expect(urlsMultiParams[1]).toBe(
    'https://example.com/models/model-00002-of-00002.gguf?param1=value1&param2=value2'
  );

  // Test with no-inline parameter (common in Vite)
  const urlWithNoInline =
    'https://example.com/models/model-00001-of-00002.gguf?no-inline';
  const urlsNoInline = ModelManager.parseModelUrl(urlWithNoInline);
  expect(urlsNoInline.length).toBe(2);
  expect(urlsNoInline[0]).toBe(
    'https://example.com/models/model-00001-of-00002.gguf?no-inline'
  );
  expect(urlsNoInline[1]).toBe(
    'https://example.com/models/model-00002-of-00002.gguf?no-inline'
  );
});

test.sequential('download split model', async () => {
  const manager = new ModelManager();
  const model = await manager.downloadModel(SPLIT_MODEL);
  expect(model.files.length).toBe(3);
  // check names
  expect(model.files[0].metadata.originalURL).toMatch(/-00001-of-00003\.gguf$/);
  expect(model.files[1].metadata.originalURL).toMatch(/-00002-of-00003\.gguf$/);
  expect(model.files[2].metadata.originalURL).toMatch(/-00003-of-00003\.gguf$/);
  // check sizes
  expect(model.files[0].size).toBe(10517152);
  expect(model.files[1].size).toBe(10381216);
  expect(model.files[2].size).toBe(5773312);
});

test.sequential('get downloaded split model', async () => {
  const manager = new ModelManager();
  const models = await manager.getModels();
  const model = models.find((m) => m.url === SPLIT_MODEL);
  expect(model).toBeDefined();
  if (!model) throw new Error();
  // check names
  expect(model.files[0].metadata.originalURL).toMatch(/-00001-of-00003\.gguf$/);
  expect(model.files[1].metadata.originalURL).toMatch(/-00002-of-00003\.gguf$/);
  expect(model.files[2].metadata.originalURL).toMatch(/-00003-of-00003\.gguf$/);
});

const SPLIT_SHARDS = ModelManager.parseModelUrl(SPLIT_MODEL);
const getRequests = (net: NetworkHook) =>
  net.requests.filter((r) => r.method === 'GET' && r.url.includes('/resolve/'));

test.sequential(
  'interrupt download split model (partial files downloaded)',
  async () => {
    // one download at a time, so the cache state at abort time is known
    const manager = new ModelManager({ parallelDownloads: 1 });
    await manager.clear();
    const net = hookNetwork();
    const stalled = net
      .on({ method: 'GET', url: SPLIT_SHARDS[2] })
      .stallAfter(1024);

    const controller = new AbortController();
    const downloadPromise = manager.downloadModel(SPLIT_MODEL, {
      signal: controller.signal,
    });
    await stalled;
    controller.abort();
    await expect(downloadPromise).rejects.toThrow('aborted');

    expect((await manager.getModels()).length).toBe(0);
    const invalidModels = await manager.getModels({ includeInvalid: true });
    expect(invalidModels.length).toBe(1);
    expect(invalidModels[0].files.length).toBe(2);
  }
);

// the test below pokes the storage directly, so pin the backend instead of the default one
const opfs = new OPFSBackend();
const opfsManager = () =>
  new ModelManager({ cacheManager: new CacheManager([opfs]) });

test.sequential('recover from missing metadata', async () => {
  const manager = opfsManager();
  await manager.clear();
  await manager.downloadModel(TINY_MODEL);

  // simulate a tab closed between the file write and the metadata write
  const fileKey = await manager.cacheManager.getNameFromURL(TINY_MODEL);
  await opfs.delete(`__metadata__${fileKey}`);
  expect((await manager.getModels()).length).toBe(0);

  const model = await manager.downloadModel(TINY_MODEL);
  expect(model.files.length).toBe(1);
  expect(model.size).toBe(1185376);
  expect((await manager.getModels()).length).toBe(1);
});

const sha256Hex = async (blob: Blob) =>
  Array.from(
    new Uint8Array(
      await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())
    )
  )
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');

// checks the cached file has the same content as the file on the server
const expectSameAsRemote = async (manager: ModelManager, url: string) => {
  const expected = await getHFFileSHA256(url, {});
  expect(expected).toBeDefined();
  const blob = await manager.cacheManager.open(url);
  expect(blob).not.toBeNull();
  expect(await sha256Hex(blob!)).toBe(expected);
};

// records the Range header of each GET to the URL, from the requests that match no earlier rule
const recordRanges = (net: NetworkHook, url: string) => {
  const ranges: (string | null)[] = [];
  net.on({ method: 'GET', url }).handle((req, next) => {
    ranges.push(req.headers.get('range'));
    return next();
  });
  return ranges;
};

test.sequential('resume download after the connection drops', async () => {
  const manager = opfsManager();
  await manager.clear();
  const net = hookNetwork();

  net.on({ method: 'GET', url: TINY_MODEL }, { times: 1 }).cutAfter(100000);
  const ranges = recordRanges(net, TINY_MODEL);

  const model = await manager.downloadModel(TINY_MODEL);
  expect(ranges).toEqual(['bytes=100000-']);
  expect(model.size).toBe(1185376);
  await expectSameAsRemote(manager, TINY_MODEL);
});

test.sequential('resume partial file in cache', async () => {
  const manager = opfsManager();
  await manager.clear();
  const net = hookNetwork();

  // connection drops after 1024 bytes, without retry
  net.on({ method: 'GET', url: TINY_MODEL }, { times: 1 }).cutAfter(1024);
  await expect(
    manager.downloadModel(TINY_MODEL, { retries: 0 })
  ).rejects.toThrow('network error');
  const fileKey = await manager.cacheManager.getNameFromURL(TINY_MODEL);
  expect(await manager.cacheManager.getSize(fileKey)).toBe(1024);
  expect((await manager.getModels()).length).toBe(0);

  const ranges = recordRanges(net, TINY_MODEL);
  let lastProgress = { loaded: 0, total: 0 };
  const model = await manager.downloadModel(TINY_MODEL, {
    progressCallback: (progress) => {
      lastProgress = progress;
    },
  });
  expect(ranges).toEqual(['bytes=1024-']);
  expect(lastProgress).toEqual({ loaded: 1185376, total: 1185376 });
  expect(model.files.length).toBe(1);
  expect(model.size).toBe(1185376);
  await expectSameAsRemote(manager, TINY_MODEL);
});

test.sequential(
  'download again if the file changed on the server',
  async () => {
    const manager = opfsManager();
    await manager.clear();
    const net = hookNetwork();

    net.on({ method: 'GET', url: TINY_MODEL }, { times: 1 }).cutAfter(1024);
    await expect(
      manager.downloadModel(TINY_MODEL, { retries: 0 })
    ).rejects.toThrow('network error');

    // the resumed response has another ETag: the partial file must not be kept
    net
      .on({ method: 'GET', url: TINY_MODEL }, { times: 1 })
      .handle(async (_req, next) => {
        const res = await next();
        const headers = new Headers(res.headers);
        headers.set('etag', '"changed"');
        return new Response(res.body, { status: res.status, headers });
      });
    const ranges = recordRanges(net, TINY_MODEL);

    const model = await manager.downloadModel(TINY_MODEL);
    expect(ranges).toEqual(['bytes=0-']); // from the start
    expect(model.size).toBe(1185376);
    await expectSameAsRemote(manager, TINY_MODEL);
  }
);

// a response with the given headers and the first `bytes` bytes of the body, then a network error
const cutResponse = async (
  res: Response,
  bytes: number,
  headers: Record<string, string>
) => {
  const data = new Uint8Array(await res.arrayBuffer()).slice(0, bytes);
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(data);
      controller.error(new TypeError('network error (simulated)'));
    },
  });
  const h = new Headers(res.headers);
  Object.entries(headers).forEach(([k, v]) => h.set(k, v));
  return new Response(body, { status: res.status, headers: h });
};

test.sequential('continue if the server sends a shorter range', async () => {
  const manager = opfsManager();
  await manager.clear();
  const net = hookNetwork();

  net.on({ method: 'GET', url: TINY_MODEL }, { times: 1 }).cutAfter(1024);
  await expect(
    manager.downloadModel(TINY_MODEL, { retries: 0 })
  ).rejects.toThrow('network error');

  // the server answers the range with bytes 1024-50000 only
  net
    .on({ method: 'GET', url: TINY_MODEL }, { times: 1 })
    .handle(async (_req, next) => {
      const res = await next();
      const part = new Uint8Array(await res.arrayBuffer()).slice(
        0,
        50001 - 1024
      );
      const headers = new Headers(res.headers);
      headers.set('content-range', 'bytes 1024-50000/1185376');
      headers.set('content-length', String(part.byteLength));
      return new Response(part, { status: 206, headers });
    });
  const ranges = recordRanges(net, TINY_MODEL);

  const model = await manager.downloadModel(TINY_MODEL);
  expect(ranges).toEqual(['bytes=50001-']);
  expect(model.size).toBe(1185376);
  await expectSameAsRemote(manager, TINY_MODEL);
});

test.sequential('do not resume with a weak ETag', async () => {
  const manager = opfsManager();
  await manager.clear();
  const net = hookNetwork();

  net
    .on({ method: 'GET', url: TINY_MODEL }, { times: 1 })
    .handle(async (_req, next) =>
      cutResponse(await next(), 1024, { etag: 'W/"weak"' })
    );
  const ranges = recordRanges(net, TINY_MODEL);

  const model = await manager.downloadModel(TINY_MODEL);
  expect(ranges).toEqual(['bytes=0-']); // from the start
  expect(model.size).toBe(1185376);
  await expectSameAsRemote(manager, TINY_MODEL);
});

// stories260K (1185376 bytes) in 5 parts
const smallParts = () => {
  const saved = { ...DOWNLOAD_CONFIG };
  DOWNLOAD_CONFIG.partsMinSize = 256 * 1024;
  DOWNLOAD_CONFIG.partSize = 256 * 1024;
  DOWNLOAD_CONFIG.savePrefixEvery = 256 * 1024;
  onTestFinished(() => Object.assign(DOWNLOAD_CONFIG, saved));
};
const partRange = (i: number) =>
  `bytes=${i * 262144}-${Math.min(1185376, (i + 1) * 262144) - 1}`;

test.sequential('download a big file in parts', async () => {
  smallParts();
  const manager = opfsManager();
  await manager.clear();
  const net = hookNetwork();
  const ranges = recordRanges(net, TINY_MODEL);

  let lastProgress = { loaded: 0, total: 0 };
  const model = await manager.downloadModel(TINY_MODEL, {
    progressCallback: (progress) => (lastProgress = progress),
  });
  // the first request probes for ranges and gives part 0
  expect(ranges[0]).toBe('bytes=0-');
  expect(ranges.slice(1).sort()).toEqual([1, 2, 3, 4].map(partRange).sort());
  expect(lastProgress).toEqual({ loaded: 1185376, total: 1185376 });
  expect(model.size).toBe(1185376);
  await expectSameAsRemote(manager, TINY_MODEL);
});

test.sequential('resume a download in parts', async () => {
  smallParts();
  const manager = opfsManager();
  await manager.clear();
  const net = hookNetwork();

  // part 1 fails, later parts may finish: a hole
  net
    .on(
      (req) =>
        req.url === TINY_MODEL && req.headers.get('range') === partRange(1),
      { times: 1 }
    )
    .cutAfter(1000);
  await expect(
    manager.downloadModel(TINY_MODEL, { retries: 0 })
  ).rejects.toThrow();
  expect((await manager.getModels()).length).toBe(0);

  // resume after part 0, not from the file size, also after a first try that fails
  net
    .on(
      (req) =>
        req.url === TINY_MODEL && req.headers.get('range') === 'bytes=262144-',
      { times: 1 }
    )
    .fail();
  const ranges = recordRanges(net, TINY_MODEL);
  const model = await manager.downloadModel(TINY_MODEL);
  expect(ranges[0]).toBe('bytes=262144-');
  expect(model.size).toBe(1185376);
  await expectSameAsRemote(manager, TINY_MODEL);
});

const isRange = (range: string) => (req: Request) =>
  req.url === TINY_MODEL && req.headers.get('range') === range;

test.sequential(
  'keep the complete part when a later try fails early',
  async () => {
    smallParts();
    const manager = opfsManager();
    await manager.clear();
    const net = hookNetwork();

    // the last part never comes, part 1 fails after parts 0, 2 and 3 are written: a hole
    void net.on(isRange(partRange(4)), { times: 1 }).stallAfter(0);
    net.on(isRange(partRange(1)), { times: 1 }).handle(async (_req, next) => {
      const res = await next();
      await new Promise((r) => setTimeout(r, 1000));
      return cutResponse(res, 1000, {});
    });
    // the next try fails before it gets to the parts
    net.on(isRange('bytes=262144-'), { times: 1 }).fail();
    const ranges = recordRanges(net, TINY_MODEL);

    const model = await manager.downloadModel(TINY_MODEL);
    // resumed after part 0, never from the end of the file
    expect(ranges).toContain('bytes=262144-');
    expect(ranges).not.toContain(`bytes=${4 * 262144}-`);
    expect(model.size).toBe(1185376);
    await expectSameAsRemote(manager, TINY_MODEL);
  }
);

test.sequential('stop the download in parts when a part fails', async () => {
  smallParts();
  const manager = opfsManager();
  await manager.clear();
  const net = hookNetwork();

  // the first response stalls, part 1 fails: the error comes out, not a stalled read
  void net.on(isRange('bytes=0-'), { times: 1 }).stallAfter(1000);
  net.on(isRange(partRange(1)), { times: 1 }).fail();
  await expect(
    manager.downloadModel(TINY_MODEL, { retries: 0 })
  ).rejects.toThrow('network error (simulated)');
});

test.sequential(
  'keep the complete part when the stream write fails to open',
  async () => {
    smallParts();
    // only the first try is big enough for parts, the retry after part 0 is a stream write
    DOWNLOAD_CONFIG.partsMinSize = 1024 * 1024;
    let failWrite = true;
    class FailOnce extends OPFSBackend {
      async write(
        key: string,
        stream: ReadableStream,
        hint?: any,
        offset?: number
      ) {
        if (failWrite && !key.startsWith('__metadata__')) {
          failWrite = false;
          throw new Error('cannot open the file (simulated)');
        }
        return super.write(key, stream, hint, offset);
      }
    }
    const manager = new ModelManager({
      cacheManager: new CacheManager([new FailOnce()]),
    });
    await manager.clear();
    const net = hookNetwork();

    // part 1 fails after the later parts are written: a hole
    void net.on(isRange(partRange(4)), { times: 1 }).stallAfter(0);
    net.on(isRange(partRange(1)), { times: 1 }).handle(async (_req, next) => {
      const res = await next();
      await new Promise((r) => setTimeout(r, 1000));
      return cutResponse(res, 1000, {});
    });
    const ranges = recordRanges(net, TINY_MODEL);

    const model = await manager.downloadModel(TINY_MODEL);
    expect(failWrite).toBe(false);
    expect(ranges.filter((r) => r === 'bytes=262144-').length).toBe(2);
    expect(model.size).toBe(1185376);
    await expectSameAsRemote(manager, TINY_MODEL);
  }
);

test.sequential('reject a part with another range', async () => {
  smallParts();
  const manager = opfsManager();
  await manager.clear();
  // the bytes of part 1, given for part 2 with the same ETag
  const real = await fetch(TINY_MODEL, { headers: { Range: partRange(1) } });
  const etag = real.headers.get('etag') ?? '';
  const part1 = await real.arrayBuffer();
  const net = hookNetwork();
  net.on(isRange(partRange(2)), { times: 1 }).handle(
    () =>
      new Response(part1, {
        status: 206,
        headers: {
          etag,
          'content-range': `bytes 262144-524287/1185376`,
          'content-length': String(part1.byteLength),
        },
      })
  );

  const model = await manager.downloadModel(TINY_MODEL);
  expect(model.size).toBe(1185376);
  await expectSameAsRemote(manager, TINY_MODEL);
});

test.sequential('give up after too many tries without new bytes', async () => {
  const manager = opfsManager();
  await manager.clear();
  const net = hookNetwork();

  net.on({ method: 'GET', url: TINY_MODEL }).fail();
  await expect(
    manager.downloadModel(TINY_MODEL, { retries: 1 })
  ).rejects.toThrow('network error');
  expect(getRequests(net).length).toBe(2);
});

test.sequential('recover from missing shard in cache', async () => {
  // one download at a time, so the cache state after the failure is known
  const manager = new ModelManager({ parallelDownloads: 1 });
  await manager.clear();
  await manager.downloadModel(TINY_MODEL);
  const net = hookNetwork();

  // connection drops during the second shard: shard 1 is complete,
  // shard 2 is partial and shard 3 is never downloaded
  net.on({ method: 'GET', url: SPLIT_SHARDS[1] }, { times: 1 }).cutAfter(1024);
  await expect(
    manager.downloadModel(SPLIT_MODEL, { retries: 0 })
  ).rejects.toThrow('network error');

  // the incomplete model does not hide the other models
  const models = await manager.getModels();
  expect(models.length).toBe(1);
  expect(models[0].url).toBe(TINY_MODEL);

  const invalidModels = await manager.getModels({ includeInvalid: true });
  const splitModel = invalidModels.find((m) => m.url === SPLIT_MODEL);
  expect(splitModel).toBeDefined();
  if (!splitModel) throw new Error();
  expect(splitModel.files.length).toBe(1);
  expect(splitModel.validate()).toBe(ModelValidationStatus.INVALID);

  let lastProgress = { loaded: 0, total: 0 };
  const tinyModel = await manager.getModelOrDownload(
    { url: TINY_MODEL },
    {
      progressCallback: (progress) => {
        lastProgress = progress;
      },
    }
  );
  expect(tinyModel.url).toBe(TINY_MODEL);
  expect(lastProgress).toEqual({
    loaded: tinyModel.size,
    total: tinyModel.size,
  });

  // only the missing shards are downloaded again
  net.reset();
  const model = await manager.getModelOrDownload({ url: SPLIT_MODEL });
  expect(model.files.length).toBe(3);
  expect(getRequests(net).map((r) => r.url)).toEqual(SPLIT_SHARDS.slice(1));
  expect((await manager.getModels()).length).toBe(2);
});

test.sequential(
  'download fails when files are missing after download',
  async () => {
    // the cache accepts the download but stores nothing
    const fakeCache = {
      download: async () => {},
      list: async () => [],
    } as unknown as CacheManager;
    const manager = new ModelManager({ cacheManager: fakeCache });
    await expect(manager.downloadModel(TINY_MODEL)).rejects.toThrow(
      'Model file not found'
    );
  }
);

test.sequential(
  'download works when the server sends no Content-Length',
  async () => {
    const manager = new ModelManager();
    await manager.clear();
    const net = hookNetwork();
    net.on({ url: TINY_MODEL }).dropHeaders('content-length');

    const model = await manager.downloadModel(TINY_MODEL);
    expect(model.files.length).toBe(1);
    expect(model.files[0].size).toBe(1185376);
    expect(model.validate()).toBe(ModelValidationStatus.VALID);

    // the cached file is reused, not downloaded again
    net.reset();
    const cached = await manager.getModelOrDownload({ url: TINY_MODEL });
    expect(cached.size).toBe(1185376);
    expect(getRequests(net).length).toBe(0);
  }
);

test.sequential(
  'download works when mmprojUrl is also a model URL',
  async () => {
    const manager = new ModelManager();
    await manager.clear();
    const net = hookNetwork();
    const model = await manager.downloadModel({
      url: TINY_MODEL,
      mmprojUrl: TINY_MODEL,
    });
    expect(model.files.length).toBe(1);
    expect(model.validate()).toBe(ModelValidationStatus.VALID);
    expect(getRequests(net).length).toBe(1);
  }
);

test.sequential(
  'keeps the order of the LoRA adapters, matches the cache on all the files',
  async () => {
    const manager = new ModelManager();
    await manager.clear();
    // any small GGUF files as stand-ins for the draft and the adapters
    const [a, b, c] = ModelManager.parseModelUrl(SPLIT_MODEL);
    const source = { url: TINY_MODEL, draftUrl: a, loraUrls: [c, b] };
    const urls = (m: Model) => m.files.map((f) => f.metadata.originalURL);
    // the order of loraUrls gives the ids of the adapters
    expect(urls(await manager.downloadModel(source))).toEqual([
      TINY_MODEL,
      a,
      c,
      b,
    ]);
    expect(urls(await manager.getModelOrDownload(source))).toEqual([
      TINY_MODEL,
      a,
      c,
      b,
    ]);
    // the draft is not listed as a model of its own
    expect((await manager.getModels()).map((m) => m.url)).toEqual([TINY_MODEL]);
    const net = hookNetwork();
    expect(urls(await manager.getModelOrDownload({ url: TINY_MODEL }))).toEqual(
      [TINY_MODEL]
    );
    expect(getRequests(net).length).toBe(0);
  }
);

test.sequential(
  'updates the extra files of a cached model without downloading it again',
  async () => {
    const manager = new ModelManager();
    await manager.clear();
    const [, b] = ModelManager.parseModelUrl(SPLIT_MODEL);
    await manager.downloadModel(TINY_MODEL);
    const net = hookNetwork();
    await manager.downloadModel({ url: TINY_MODEL, loraUrls: [b] });
    const listed = await manager.getModels();
    expect(listed.map((m) => [m.url, m.loraUrls])).toEqual([[TINY_MODEL, [b]]]);
    expect(getRequests(net).filter((r) => r.url === TINY_MODEL).length).toBe(0);
  }
);

test.sequential('download invalid model URL', async () => {
  const manager = new ModelManager();
  const invalidUrl = 'https://invalid.example.com/model.gguf';
  await expect(manager.downloadModel(invalidUrl)).rejects.toThrow();
});

test.sequential('download with abort signal', async () => {
  const manager = new ModelManager();
  await manager.clear();
  const net = hookNetwork();
  const stalled = net.on({ method: 'GET', url: TINY_MODEL }).stallAfter(1024);

  const controller = new AbortController();
  const downloadPromise = manager.downloadModel(TINY_MODEL, {
    signal: controller.signal,
  });
  await stalled;
  controller.abort();
  await expect(downloadPromise).rejects.toThrow('aborted');
  expect((await manager.getModels()).length).toBe(0);
});

test.sequential('download with progress callback', async () => {
  const manager = new ModelManager();
  await manager.clear();

  let progressCalled = false;
  let lastLoaded = 0;
  const model = await manager.downloadModel(TINY_MODEL, {
    progressCallback: ({ loaded, total }) => {
      expect(loaded).toBeGreaterThan(0);
      expect(total).toBeGreaterThan(0);
      expect(loaded).toBeLessThanOrEqual(total);
      expect(loaded).toBeGreaterThanOrEqual(lastLoaded);
      progressCalled = true;
      lastLoaded = loaded;
    },
  });

  expect(progressCalled).toBe(true);
  expect(model).toBeDefined();
  expect(model.size).toBeGreaterThan(0);
});

test.sequential('model validation status for new model', async () => {
  const manager = new ModelManager();
  const model = new Model(manager, TINY_MODEL);
  const status = await model.validate();
  expect(status).toBe(ModelValidationStatus.INVALID);
});

test.sequential('downloadModel throws on invalid URL', async () => {
  const manager = new ModelManager();
  await expect(manager.downloadModel('invalid.txt')).rejects.toThrow();
});

test.sequential('model size calculation', async () => {
  const manager = new ModelManager();
  const model = await manager.downloadModel(TINY_MODEL);
  expect(model.size).toBe(1185376);
});

test.sequential('remove model from cache', async () => {
  const manager = new ModelManager();
  await manager.clear();

  // Download model first
  const model = await manager.downloadModel(TINY_MODEL);
  expect((await manager.getModels()).length).toBe(1);
  expect(model.size).toBeGreaterThan(0);

  // Remove model
  await model.remove();
  expect(model.size).toBe(-1);

  // Try to open removed model
  await expect(model.open()).rejects.toThrow('deleted from the cache');

  // Validate removed model
  const status = await model.validate();
  expect(status).toBe(ModelValidationStatus.DELETED);

  // Cannot see it in list of models
  const models = await manager.getModels();
  expect(models.find((m) => m.url === TINY_MODEL)).toBeUndefined();
});

test.sequential('clear model manager', async () => {
  const manager = new ModelManager();
  const model = await manager.downloadModel(TINY_MODEL);
  expect(model).toBeDefined();
  expect((await manager.getModels()).length).toBeGreaterThan(0);
  await manager.clear();
  expect((await manager.getModels()).length).toBe(0);
});
