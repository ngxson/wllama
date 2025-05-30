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
export const bufToText = (buffer: ArrayBuffer | Uint8Array): string => {
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

export interface ShardInfo {
  baseURL: string;
  current: number;
  total: number;
}

const URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;

/**
 * Parse shard number and total from a file name or URL
 */
export const parseShardNumber = (fnameOrUrl: string): ShardInfo => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1,
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ''),
      current: parseInt(matches[1]),
      total: parseInt(matches[2]),
    };
  }
};

/**
 * Parses a model URL and returns an array of URLs based on the following patterns:
 * - If the input URL is an array, it returns the array itself.
 * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
 * - Otherwise, it returns an array containing the input URL as a single element array.
 * @param modelUrl URL or list of URLs
 */
export const parseModelUrl = (modelUrl: string): string[] => {
  const { baseURL, current, total } = parseShardNumber(modelUrl);
  if (current == total && total == 1) {
    return [modelUrl];
  } else {
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = queryMatch?.[1] ?? '';
    const paddedShardIds = Array.from({ length: total }, (_, index) =>
      (index + 1).toString().padStart(5, '0')
    );
    return paddedShardIds.map(
      (current) =>
        `${baseURL}-${current}-of-${total.toString().padStart(5, '0')}.gguf${queryParams}`
    );
  }
};

/**
 * Check if the given blobs are files or not, then sort them by shard number
 */
export const sortFileByShard = (blobs: Blob[]): void => {
  const isFiles = blobs.every((b) => !!(b as File).name);
  if (isFiles && blobs.length > 1) {
    const files = blobs as File[];
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
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

export const isString = (value: any): boolean => !!value?.startsWith;

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
 * Regular expression to validate GGUF file paths/URLs
 * Matches paths ending with .gguf and optional query parameters
 */
export const GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;

/**
 * Validates if a given string is a valid GGUF file path/URL
 * @param path The file path or URL to validate
 * @returns true if the path is a valid GGUF file path/URL
 */
export const isValidGgufFile = (path: string): boolean => {
  return GGUF_FILE_REGEX.test(path);
};

/**
 * Check if browser is Safari iOS / iPad / iPhone
 * Source: https://github.com/DamonOehlman/detect-browser/blob/master/src/index.ts
 */
export const isSafariMobile = (): boolean => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/); // ios
};

/**
 * Create a worker from a string
 */
export const createWorker = (workerCode: string | Blob): Worker => {
  const workerURL = URL.createObjectURL(
    isString(workerCode)
      ? new Blob([workerCode], { type: 'text/javascript' })
      : (workerCode as Blob)
  );
  return new Worker(workerURL, { type: 'module' });
};

/**
 * Convert callback to async iterator
 */
export const cbToAsyncIter =
  <A extends any[], T>(
    fn: (
      ...args: [...args: A, callback: (val?: T, done?: boolean) => void]
    ) => void
  ) =>
  (...args: A): AsyncIterable<T> => {
    let values: Promise<[T, boolean]>[] = [];
    let resolve: (x: [T, boolean]) => void;
    values.push(
      new Promise((r) => {
        resolve = r;
      })
    );
    fn(...args, (val?: T, done?: boolean) => {
      resolve([val!, done!]);
      values.push(
        new Promise((r) => {
          resolve = r;
        })
      );
    });
    return (async function* () {
      let val: T;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = await values[i];
        delete values[i];
        if (val !== undefined) yield val;
      }
    })();
  };
