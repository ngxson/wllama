import { test, expect } from 'vitest';
import { ModelManager, Model, ModelValidationStatus } from './model-manager';
import { OPFSBackend } from './storage/opfs';
import CacheManager from './cache-manager';
import { hookNetwork, type NetworkHook } from './testing/network-hook';

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

test.sequential('recover from partial file in cache', async () => {
  const manager = new ModelManager();
  await manager.clear();
  const net = hookNetwork();

  // connection drops after 1024 bytes
  net.on({ method: 'GET', url: TINY_MODEL }, { times: 1 }).cutAfter(1024);
  await expect(manager.downloadModel(TINY_MODEL)).rejects.toThrow(
    'network error'
  );
  const fileKey = await manager.cacheManager.getNameFromURL(TINY_MODEL);
  expect(await manager.cacheManager.getSize(fileKey)).toBe(1024);
  expect((await manager.getModels()).length).toBe(0);

  const model = await manager.downloadModel(TINY_MODEL);
  expect(model.files.length).toBe(1);
  expect(model.files[0].size).toBe(1185376);
  expect(model.size).toBe(1185376);
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
  await expect(manager.downloadModel(SPLIT_MODEL)).rejects.toThrow(
    'network error'
  );

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
