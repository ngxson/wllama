import { expect, test, describe } from 'vitest';
import {
  joinBuffers,
  bufToText,
  padDigits,
  sumArr,
  isString,
  delay,
  absoluteUrl,
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
