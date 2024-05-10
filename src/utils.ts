type ProgressCallback = (opts: { loaded: number, total: number }) => any;

const _loadBinaryResource = async (
  url: string,
  progressCallback?: ProgressCallback,
): Promise<ArrayBuffer> => {
  const reportProgress = progressCallback && typeof progressCallback === 'function';
  // @ts-ignore
  const window = self;
  const isInsideBrowser = (typeof window !== 'undefined');
  let cache: Cache | null = null;

  // Try to find if the model data is cached in Web Worker memory.
  // @ts-ignore
  if (isInsideBrowser && window.caches) {
    // @ts-ignore
    cache = await window.caches.open('wllama_cache');
    // @ts-ignore
    const cachedResponse = await cache.match(url);

    if (cachedResponse) {
      const data = await cachedResponse.arrayBuffer();
      if (reportProgress) {
        progressCallback({
          total: data.byteLength,
          loaded: data.byteLength,
        })
      }
      return data;
    }
  }


  // Download model and store in cache
  const _promise = new Promise<ArrayBuffer>((resolve, reject) => {
    const req = new XMLHttpRequest();
    req.open('GET', url, true);
    req.responseType = 'arraybuffer';
    req.onload = async (_) => {
      const arrayBuffer = req.response; // Note: not req.responseText
      if (arrayBuffer) {
        if (cache) {
          await cache.put(url, new Response(arrayBuffer));
        };
        resolve(arrayBuffer);
      }
    };
    if (reportProgress) {
      req.onprogress = progressCallback;
    }
    req.onerror = (err) => {
      reject(err);
    };
    req.send(null);
  });

  return await _promise;
}

/**
 * Return file size in bytes
 */
export const getBinarySize = (url: string): Promise<number> => {
  return new Promise<number>((resolve, reject) => {
    const req = new XMLHttpRequest();
    req.open('GET', url, true);
    req.onreadystatechange = () => {
      if (req.readyState === 2) { // HEADERS_RECEIVED
        const value = req.getResponseHeader('Content-Length') ?? '';
        req.abort();
        if (isNaN(+value)) {
          reject(new Error('Content-Length is not a number'));
        } else {
          resolve(parseInt(value));
        }
      }
    };
    req.onerror = (err) => {
      reject(err);
    };
    req.send(null);
  });
};

export const joinBuffers = (buffers: Uint8Array[]): Uint8Array => {
  const totalSize = buffers.reduce((acc, buf) => acc + buf.length, 0);
  const output = new Uint8Array(totalSize);
  output.set(buffers[0], 0);
  for (let i = 1; i < buffers.length; i++) {
    output.set(buffers[i], buffers[i - 1].length);
  }
  return output;
};

/**
 * Load a resource as byte array. If multiple URLs is given, we will assume that the resource is splitted into small files
 * @param url URL (or list of URLs) to resource
 */
export const loadBinaryResource = async (
  url: string | string[],
  nMaxParallel: number,
  progressCallback?: (opts: { loaded: number, total: number }) => any,
): Promise<ArrayBuffer | ArrayBuffer[]> => {
  const reportProgress = progressCallback && typeof progressCallback === 'function';
  const urls: string[] = Array.isArray(url)
    ? [...url] as string[]
    : [url as string];

  const tasks: {
    url: string,
    result: ArrayBuffer,
    started: boolean,
    sizeTotal: number,
    sizeLoaded: number,
  }[] = urls.map(u => ({
    url: u,
    result: new ArrayBuffer(0),
    started: false,
    sizeTotal: 0,
    sizeLoaded: 0,
  }));

  // Get the total length of all files
  let totalSize = -1;
  if (reportProgress) {
    const sizeArr = await Promise.all(urls.map(u => getBinarySize(u)));
    totalSize = sumArr(sizeArr);
  }

  // This is not multi-thread, but just a simple naming to borrow the idea
  const threads: Promise<void>[] = [];
  const runDownloadThread = async () => {
    while (true) {
      const task = tasks.find(t => !t.started);
      if (!task) return;
      task.started = true;
      task.result = await _loadBinaryResource(task.url, (progress) => {
        task.sizeTotal = progress.total;
        task.sizeLoaded = progress.loaded;
        if (reportProgress) {
          // report aggregated progress across all files
          progressCallback({
            loaded: sumArr(tasks.map(t => t.sizeLoaded)),
            total: totalSize,
          });
        }
      });
    }
  };
  for (let i = 0; i < nMaxParallel; i++) {
    threads.push(runDownloadThread());
  }
  // wait until all downloads finish
  await Promise.all(threads);

  return tasks.length === 1
    ? tasks[0].result
    : tasks.map(r => r.result);
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
export const getWModuleConfig = (pathConfig: { [filename: string]: string }) => {
  return {
    noInitialRun: true,
    print: function(text: any) {
      if (arguments.length > 1) text = Array.prototype.slice.call(arguments).join(' ');
      console.log(text);
    },
    printErr: function(text: any) {
      if (arguments.length > 1) text = Array.prototype.slice.call(arguments).join(' ');
      console.warn(text);
    },
    // @ts-ignore
    locateFile: function (filename: string, basePath: string) {
      const p = pathConfig[filename];
      console.log(`Loading "${filename}" from "${p}"`);
      return p;
    },
  };
}

export const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

export const absoluteUrl = (relativePath: string) => new URL(relativePath, document.baseURI).href;

export const padDigits = (number: number, digits: number) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join('0') + number;
}

export const sumArr = (arr: number[]) => arr.reduce((prev, curr) => prev + curr, 0);

/**
 * Browser feature detection
 * Copied from https://unpkg.com/wasm-feature-detect?module (Apache License)
 */

/**
 * @returns true if browser support multi-threads
 */
export const isSupportMultiThread = () => (async e => {try {return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);} catch (e) {return !1;}})(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5, 4, 1, 3, 1, 1, 10, 11, 1, 9, 0, 65, 0, 254, 16, 2, 0, 26, 11]));

/**
 * @returns true if browser support wasm "native" exception handler
 */
const isSupportExceptions = async () => WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 8, 1, 6, 0, 6, 64, 25, 11, 11]));

/**
 * @returns true if browser support wasm SIMD
 */
const isSupportSIMD = async () => WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));

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
