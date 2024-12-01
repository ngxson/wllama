import { test, expect } from 'vitest';
import { ModelManager, Model, ModelValidationStatus } from './model-manager';

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf';
const SPLIT_MODEL =
  'https://huggingface.co/ngxson/tinyllama_split_test/resolve/main/stories15M-q8_0-00001-of-00003.gguf';

test('parseModelUrl handles single model URL', () => {
  const urls = ModelManager.parseModelUrl(TINY_MODEL);
  expect(urls).toEqual([TINY_MODEL]);
});

test('parseModelUrl handles array of URLs', () => {
  const urls = ModelManager.parseModelUrl(SPLIT_MODEL);
  expect(urls.length).toBe(3);
  expect(urls[0]).toMatch(/-00001-of-00003\.gguf$/);
  expect(urls[1]).toMatch(/-00002-of-00003\.gguf$/);
  expect(urls[2]).toMatch(/-00003-of-00003\.gguf$/);
});

test('download split model', async () => {
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

test('download invalid model URL', async () => {
  const manager = new ModelManager();
  const invalidUrl = 'https://invalid.example.com/model.gguf';
  await expect(manager.downloadModel(invalidUrl)).rejects.toThrow();
});

test('clear model manager', async () => {
  const manager = new ModelManager();
  const model = await manager.downloadModel(TINY_MODEL);
  expect(model).toBeDefined();
  expect((await manager.getModels()).length).toBeGreaterThan(0);
  await manager.clear();
  expect((await manager.getModels()).length).toBe(0);
});

test('download with abort signal', async () => {
  const manager = new ModelManager();
  await manager.clear();
  const controller = new AbortController();
  const downloadPromise = manager.downloadModel(TINY_MODEL, {
    signal: controller.signal,
  });
  setTimeout(() => controller.abort(), 10);
  await downloadPromise.catch(console.error);
  await expect(downloadPromise).rejects.toThrow('aborted');
});

test('model validation status for new model', async () => {
  const manager = new ModelManager();
  const model = new Model(manager, TINY_MODEL);
  const status = await model.validate();
  expect(status).toBe(ModelValidationStatus.INVALID);
});

test('downloadModel throws on invalid URL', async () => {
  const manager = new ModelManager();
  await expect(manager.downloadModel('invalid.txt')).rejects.toThrow();
});

test('model size calculation', async () => {
  const manager = new ModelManager();
  const model = await manager.downloadModel(TINY_MODEL);
  expect(model.size).toBe(1185376);
});
