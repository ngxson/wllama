import { expect, test, describe } from 'vitest';
import {
  joinBuffers,
  bufToText,
  padDigits,
  sumArr,
  isString,
  delay,
  absoluteUrl,
  parseShardNumber,
  parseModelUrl,
  sortFileByShard,
  isValidGgufFile,
} from './utils';

describe('joinBuffers', () => {
  test('joins multiple buffers correctly', () => {
    const buf1 = new Uint8Array([1, 2, 3]);
    const buf2 = new Uint8Array([4, 5]);

    const result = joinBuffers([buf1, buf2]);

    expect(result).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
  });
});

describe('bufToText', () => {
  test('converts buffer to string', () => {
    const buffer = new TextEncoder().encode('hello world');
    expect(bufToText(buffer)).toBe('hello world');
  });
});

describe('padDigits', () => {
  test('pads number with zeros', () => {
    expect(padDigits(5, 3)).toBe('005');
    expect(padDigits(42, 4)).toBe('0042');
    expect(padDigits(1234, 2)).toBe('1234');
  });
});

describe('sumArr', () => {
  test('sums array of numbers', () => {
    expect(sumArr([1, 2, 3])).toBe(6);
    expect(sumArr([])).toBe(0);
    expect(sumArr([5])).toBe(5);
  });
});

describe('isString', () => {
  test('checks if value is string', () => {
    expect(isString('test')).toBe(true);
    expect(isString('')).toBe(true);
    expect(isString(null)).toBe(false);
    expect(isString(undefined)).toBe(false);
    expect(isString(123)).toBe(false);
  });
});

describe('delay', () => {
  test('delays execution', async () => {
    const start = Date.now();
    await delay(100);
    const elapsed = Date.now() - start;
    expect(elapsed).toBeGreaterThanOrEqual(100);
  });
});

describe('absoluteUrl', () => {
  test('converts relative path to absolute url', () => {
    // Mock document.baseURI
    Object.defineProperty(document, 'baseURI', {
      value: 'http://example.com/app/',
      writable: true,
    });

    expect(absoluteUrl('test.html')).toBe('http://example.com/app/test.html');
    expect(absoluteUrl('/test.html')).toBe('http://example.com/test.html');
  });
});

describe('shard processing', () => {
  test('parseShardNumber extracts correct info', () => {
    expect(parseShardNumber('abcdef-123456-00001-of-00005.gguf')).toEqual({
      baseURL: 'abcdef-123456',
      current: 1,
      total: 5,
    });

    expect(parseShardNumber('abcdef-123456.9090-q8_0.gguf')).toEqual({
      baseURL: 'abcdef-123456.9090-q8_0.gguf',
      current: 1,
      total: 1,
    });

    expect(
      parseShardNumber('abcdef-123456-00001-of-00005.gguf?param=value')
    ).toEqual({
      baseURL: 'abcdef-123456',
      current: 1,
      total: 5,
    });

    expect(
      parseShardNumber('abcdef-123456-00001-of-00005.gguf?no-inline')
    ).toEqual({
      baseURL: 'abcdef-123456',
      current: 1,
      total: 5,
    });

    expect(
      parseShardNumber(
        'abcdef-123456-00001-of-00005.gguf?param1=value1&param2=value2'
      )
    ).toEqual({
      baseURL: 'abcdef-123456',
      current: 1,
      total: 5,
    });
  });

  test('parseModelUrl generates correct shard URLs', () => {
    const singleFile = 'model.gguf';
    expect(parseModelUrl(singleFile)).toEqual(['model.gguf']);

    const shardedFile = 'model-00001-of-00003.gguf';
    expect(parseModelUrl(shardedFile)).toEqual([
      'model-00001-of-00003.gguf',
      'model-00002-of-00003.gguf',
      'model-00003-of-00003.gguf',
    ]);

    const complexPath = 'https://example.com/models/llama-00001-of-00002.gguf';
    expect(parseModelUrl(complexPath)).toEqual([
      'https://example.com/models/llama-00001-of-00002.gguf',
      'https://example.com/models/llama-00002-of-00002.gguf',
    ]);

    const shardedFileWithQuery = 'model-00001-of-00003.gguf?param=value';
    expect(parseModelUrl(shardedFileWithQuery)).toEqual([
      'model-00001-of-00003.gguf?param=value',
      'model-00002-of-00003.gguf?param=value',
      'model-00003-of-00003.gguf?param=value',
    ]);

    const complexPathWithQuery =
      'https://example.com/models/llama-00001-of-00002.gguf?no-inline';
    expect(parseModelUrl(complexPathWithQuery)).toEqual([
      'https://example.com/models/llama-00001-of-00002.gguf?no-inline',
      'https://example.com/models/llama-00002-of-00002.gguf?no-inline',
    ]);
  });

  test('sortFileByShard sorts files by shard number', () => {
    const files = [
      new File(
        [],
        'e2fc714c4727ee9395f324cd2e7f331f-model-00003-of-00005.gguf'
      ),
      new File(
        [],
        '187ef4436122d1cc2f40dc2b92f0eba0-model-00001-of-00005.gguf'
      ),
      new File(
        [],
        'c4357687ea2b461cb07cf0a0a3de939f-model-00002-of-00005.gguf'
      ),
      new File(
        [],
        '6a4d40512eabd63221cbdf3df4636cd7-model-00005-of-00005.gguf'
      ),
      new File(
        [],
        '0952e4c6ba320f5278605eb5333eec0f-model-00004-of-00005.gguf'
      ),
    ];

    sortFileByShard(files);

    expect(files.map((f) => parseShardNumber(f.name).current)).toEqual([
      1, 2, 3, 4, 5,
    ]);

    // Single file should not be affected
    const singleFile = [new File([], 'model.gguf')];
    sortFileByShard(singleFile);
    expect(singleFile[0].name).toBe('model.gguf');

    // Regular blobs should not be affected
    const blobs = [new Blob(), new Blob()];
    sortFileByShard(blobs);
    expect(blobs.length).toBe(2);
  });
});

describe('GGUF file validation', () => {
  test('isValidGgufFile should correctly validate GGUF files', () => {
    // Basic valid cases
    expect(isValidGgufFile('model.gguf')).toBe(true);
    expect(isValidGgufFile('path/to/model.gguf')).toBe(true);

    // With Vite query parameters
    expect(isValidGgufFile('model.gguf?no-inline')).toBe(true);
    expect(isValidGgufFile('foo.gguf?no-inline')).toBe(true);

    // With query parameters
    expect(isValidGgufFile('model.gguf?param=value')).toBe(true);
    expect(isValidGgufFile('model.gguf?param1&param2')).toBe(true);
    expect(isValidGgufFile('model.gguf?no-inline&v=123')).toBe(true);
    expect(isValidGgufFile('model.gguf?param1=value1&param2=value2')).toBe(
      true
    );
    expect(isValidGgufFile('model.gguf?param=value&special=!@#$%^&*()')).toBe(
      true
    );
    expect(isValidGgufFile('model.gguf?')).toBe(true);

    // With fragments
    expect(isValidGgufFile('model.gguf?param=value#fragment')).toBe(true);

    // Invalid cases
    expect(isValidGgufFile('model.bin')).toBe(false);
    expect(isValidGgufFile('model.gguf.bin')).toBe(false);
    expect(isValidGgufFile('modelgguf')).toBe(false);
    expect(isValidGgufFile('path/to/model.txt?ext=gguf')).toBe(false);

    // Case sensitivity
    expect(isValidGgufFile('model.GGUF?param=value')).toBe(false);
  });
});
