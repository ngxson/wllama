import { test, expect } from 'vitest';
import { ModelManager, Model, ModelValidationStatus } from './model-manager';

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

// skip on CI, only run locally with a slow connection
test.skip('interrupt download split model (partial files downloaded)', async () => {
  const manager = new ModelManager();
  await manager.clear();
  const controller = new AbortController();
  const downloadPromise = manager.downloadModel(SPLIT_MODEL, {
    signal: controller.signal,
    progressCallback: ({ loaded, total }) => {
      const progress = loaded / total;
      if (progress > 0.8) {
        controller.abort();
      }
    },
  });
  await expect(downloadPromise).rejects.toThrow('aborted');
  expect((await manager.getModels()).length).toBe(0);
  expect((await manager.getModels({ includeInvalid: true })).length).toBe(1);
});

test.sequential('download invalid model URL', async () => {
  const manager = new ModelManager();
  const invalidUrl = 'https://invalid.example.com/model.gguf';
  await expect(manager.downloadModel(invalidUrl)).rejects.toThrow();
});

test.sequential('download with abort signal', async () => {
  const manager = new ModelManager();
  await manager.clear();
  const controller = new AbortController();
  const downloadPromise = manager.downloadModel(TINY_MODEL, {
    signal: controller.signal,
  });
  setTimeout(() => controller.abort(), 10);
  await downloadPromise.catch(console.error);
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

test.sequential('import local gguf file into cache', async () => {
  const manager = new ModelManager();
  await manager.clear();

  const fileBytes = new Uint8Array([
    0x47, 0x47, 0x55, 0x46, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0x20, 0x00, 0x00, 0x00,
  ]);
  const file = new File([fileBytes], 'tiny-local.gguf', {
    type: 'application/octet-stream',
  });

  let lastProgress: { loaded: number; total: number } | undefined;
  const model = await manager.importFile(file, (progress) => {
    lastProgress = progress;
  });

  expect(model.url).toBe('local://tiny-local.gguf');
  expect(model.size).toBe(file.size);
  expect(model.validate()).toBe(ModelValidationStatus.VALID);
  expect(lastProgress).toEqual({ loaded: file.size, total: file.size });

  const cachedModels = await manager.getModels();
  const cachedModel = cachedModels.find((m) => m.url === model.url);
  expect(cachedModel).toBeDefined();

  const importedAgain = await manager.importFile(file);
  expect(importedAgain.url).toBe(model.url);
  const cachedModelsAfterSecondImport = await manager.getModels();
  expect(
    cachedModelsAfterSecondImport.filter((m) => m.url === model.url)
  ).toHaveLength(1);

  const [blob] = await cachedModel!.open();
  const importedBytes = new Uint8Array(await blob.arrayBuffer());
  expect(Array.from(importedBytes)).toEqual(Array.from(fileBytes));
});

test.sequential('importFile rejects non-gguf extension', async () => {
  const manager = new ModelManager();
  const file = new File(['not a gguf'], 'tiny-local.bin', {
    type: 'application/octet-stream',
  });

  await expect(manager.importFile(file)).rejects.toThrow('must end with ".gguf"');
});

test.sequential('importFile rejects invalid gguf magic header', async () => {
  const manager = new ModelManager();
  const file = new File([new Uint8Array([0x00, 0x01, 0x02, 0x03])], 'bad.gguf', {
    type: 'application/octet-stream',
  });

  await expect(manager.importFile(file)).rejects.toThrow(
    'does not start with GGUF magic number'
  );
});

test.sequential('importFile overwrites same-name local model with new contents', async () => {
  const manager = new ModelManager();
  await manager.clear();

  const firstBytes = new Uint8Array([
    0x47, 0x47, 0x55, 0x46, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0x20, 0x00, 0x00, 0x00,
  ]);
  const secondBytes = new Uint8Array([
    0x47, 0x47, 0x55, 0x46, 0xaa, 0xbb, 0xcc, 0xdd, 0x99, 0x88, 0x77, 0x66,
    0x55, 0x44, 0x33, 0x22,
  ]);

  const firstFile = new File([firstBytes], 'replace-me.gguf', {
    type: 'application/octet-stream',
  });
  const secondFile = new File([secondBytes], 'replace-me.gguf', {
    type: 'application/octet-stream',
  });

  await manager.importFile(firstFile);
  await manager.importFile(secondFile);

  const models = await manager.getModels();
  expect(models.filter((m) => m.url === 'local://replace-me.gguf')).toHaveLength(
    1
  );

  const model = models.find((m) => m.url === 'local://replace-me.gguf');
  expect(model).toBeDefined();

  const [blob] = await model!.open();
  const bytes = new Uint8Array(await blob.arrayBuffer());
  expect(Array.from(bytes)).toEqual(Array.from(secondBytes));
});
