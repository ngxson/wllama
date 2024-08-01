export const joinBuffers = (buffers: Uint8Array[]): Uint8Array => {
  const totalSize = buffers.reduce((acc, buf) => acc + buf.length, 0);
  const output = new Uint8Array(totalSize);
  output.set(buffers[0], 0);
  for (let i = 1; i < buffers.length; i++) {
    output.set(buffers[i], buffers[i - 1].length);
  }
  return output;
};

const textDecoder = new TextDecoder();

/**
 * Convert list of bytes (number) to text
 * @param buffer
 * @returns a string
 */
export const bufToText = (buffer: ArrayBuffer): string => {
  return textDecoder.decode(buffer);
};

/**
 * Get default stdout/stderr config for wasm module
 */
export const getWModuleConfig = (pathConfig: {
  [filename: string]: string;
}) => {
  return {
    noInitialRun: true,
    print: function (text: any) {
      if (arguments.length > 1)
        text = Array.prototype.slice.call(arguments).join(' ');
      console.log(text);
    },
    printErr: function (text: any) {
      if (arguments.length > 1)
        text = Array.prototype.slice.call(arguments).join(' ');
      console.warn(text);
    },
    // @ts-ignore
    locateFile: function (filename: string, basePath: string) {
      const p = pathConfig[filename];
      console.log(`Loading "${filename}" from "${p}"`);
      return p;
    },
  };
};

/**
 * Check if the given blobs are files or not, then sort them by name
 */
export const maybeSortFileByName = (blobs: Blob[]): void => {
  const isFiles = blobs.every((b) => !!(b as File).name);
  if (isFiles) {
    const files = blobs as File[];
    files.sort((a, b) => a.name.localeCompare(b.name));
  }
};

export const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

export const absoluteUrl = (relativePath: string) =>
  new URL(relativePath, document.baseURI).href;

export const padDigits = (number: number, digits: number) => {
  return (
    Array(Math.max(digits - String(number).length + 1, 0)).join('0') + number
  );
};

export const sumArr = (arr: number[]) =>
  arr.reduce((prev, curr) => prev + curr, 0);

/**
 * Browser feature detection
 * Copied from https://unpkg.com/wasm-feature-detect?module (Apache License)
 */

/**
 * @returns true if browser support multi-threads
 */
export const isSupportMultiThread = () =>
  (async (e) => {
    try {
      return (
        'undefined' != typeof MessageChannel &&
          new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)),
        WebAssembly.validate(e)
      );
    } catch (e) {
      return !1;
    }
  })(
    new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5, 4, 1, 3, 1,
      1, 10, 11, 1, 9, 0, 65, 0, 254, 16, 2, 0, 26, 11,
    ])
  );

/**
 * @returns true if browser support wasm "native" exception handler
 */
const isSupportExceptions = async () =>
  WebAssembly.validate(
    new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 8, 1, 6,
      0, 6, 64, 25, 11, 11,
    ])
  );

/**
 * @returns true if browser support wasm SIMD
 */
const isSupportSIMD = async () =>
  WebAssembly.validate(
    new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10,
      1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
    ])
  );

/**
 * Throws an error if the environment is not compatible
 */
export const checkEnvironmentCompatible = async (): Promise<void> => {
  if (!(await isSupportExceptions())) {
    throw new Error('WebAssembly runtime does not support exception handling');
  }
  if (!(await isSupportSIMD())) {
    throw new Error('WebAssembly runtime does not support SIMD');
  }
};

/**
 * Check if browser is Safari
 * Source: https://github.com/DamonOehlman/detect-browser/blob/master/src/index.ts
 */
export const isSafari = (): boolean => {
  return (
    isSafariMobile() ||
    !!navigator.userAgent.match(/Version\/([0-9\._]+).*Safari/)
  ); // safari
};

/**
 * Check if browser is Safari iOS / iPad / iPhone
 * Source: https://github.com/DamonOehlman/detect-browser/blob/master/src/index.ts
 */
export const isSafariMobile = (): boolean => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/); // ios
};
