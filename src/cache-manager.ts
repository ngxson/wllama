import type { DownloadProgressCallback } from './model-manager';
import { createWorker, isSafariMobile } from './utils';
import { OPFS_UTILS_WORKER_CODE } from './workers-code/generated';

const PREFIX_METADATA = '__metadata__';

export type DownloadOptions = {
  /**
   * Callback function to track download progress
   */
  progressCallback?: DownloadProgressCallback;
  /**
   * Custom headers for the request. Useful for authentication (e.g. Bearer token)
   */
  headers?: Record<string, string>;
  /**
   * Abort signal for the request
   */
  signal?: AbortSignal;
};

// To prevent breaking change, we fill etag with a pre-defined value
export const POLYFILL_ETAG = 'polyfill_for_older_version';

export interface CacheEntry {
  /**
   * File name in OPFS, in the format: `${hashSHA1(fullURL)}_${fileName}`
   */
  name: string;
  /**
   * Size of file (in bytes)
   */
  size: number;
  /**
   * Other metadata
   */
  metadata: CacheEntryMetadata;
}

export interface CacheEntryMetadata {
  /**
   * ETag header from remote request
   * https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
   */
  etag: string;
  /**
   * Remote file size (in bytes), used for integrity check
   */
  originalSize: number;
  /**
   * Original URL of the remote model. Unused for now
   */
  originalURL: string;
}

/**
 * Cache implementation using OPFS (Origin private file system)
 *
 * This class is also responsible for downloading files from the internet.
 */
class CacheManager {
  /**
   * Convert a given URL into file name in cache.
   *
   * Format of the file name: `${hashSHA1(fullURL)}_${fileName}`
   */
  async getNameFromURL(url: string): Promise<string> {
    return await urlToFileName(url, '');
  }

  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  async write(
    name: string,
    stream: ReadableStream,
    metadata: CacheEntryMetadata
  ): Promise<void> {
    this.writeMetadata(name, metadata); // no need await
    return await opfsWrite(name, stream);
  }

  async download(url: string, options: DownloadOptions = {}): Promise<void> {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let aborted = false;
    if (options.signal) {
      aborted = options.signal.aborted;
      const mSignal = options.signal;
      mSignal.addEventListener('abort', () => {
        aborted = true;
        worker.postMessage({ action: 'download-abort' });
      });
      delete options.signal;
    }
    const metadataFileName: string = await urlToFileName(url, PREFIX_METADATA);
    const filename: string = await urlToFileName(url, '');
    return await new Promise((resolve, reject) => {
      worker.postMessage({
        action: 'download',
        url,
        filename,
        metadataFileName,
        options: { headers: options.headers, aborted },
      });
      worker.onmessage = (e: MessageEvent<any>) => {
        if (e.data.ok) {
          worker.terminate();
          resolve();
        } else if (e.data.err) {
          worker.terminate();
          reject(e.data.err);
        } else if (e.data.progress) {
          const progress: { loaded: number; total: number } = e.data.progress;
          options.progressCallback?.(progress);
        } else {
          // should never happen
          reject(new Error('Unknown message from worker'));
          console.error('Unknown message from worker', e.data);
        }
      };
    });
  }

  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  async open(nameOrURL: string): Promise<Blob | null> {
    return await opfsOpen(nameOrURL);
  }

  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  async getSize(name: string): Promise<number> {
    return await opfsFileSize(name);
  }

  /**
   * Get metadata of a cached file
   */
  async getMetadata(name: string): Promise<CacheEntryMetadata | null> {
    const stream = await opfsOpen(name, PREFIX_METADATA);
    const cachedSize = await this.getSize(name);
    if (!stream) {
      return cachedSize > 0
        ? // files created by older version of wllama doesn't have metadata, we will try to polyfill it
          {
            etag: POLYFILL_ETAG,
            originalSize: cachedSize,
            originalURL: '',
          }
        : // if cached file not found, we don't have metadata at all
          null;
    }
    try {
      const meta = await new Response(stream).json();
      return meta;
    } catch (e) {
      // worst case: metadata is somehow corrupted, we will re-download the model
      return null;
    }
  }

  /**
   * List all files currently in cache
   */
  async list(): Promise<CacheEntry[]> {
    const cacheDir = await getCacheDir();
    const result: CacheEntry[] = [];
    const metadataMap: Record<string, CacheEntryMetadata> = {};
    // @ts-ignore
    for await (let [name, handler] of cacheDir.entries()) {
      if (handler.kind === 'file' && name.startsWith(PREFIX_METADATA)) {
        const stream = (
          await (handler as FileSystemFileHandle).getFile()
        ).stream();
        const meta = await new Response(stream).json().catch((_) => null);
        metadataMap[name.replace(PREFIX_METADATA, '')] = meta;
      }
    }
    // @ts-ignore
    for await (let [name, handler] of cacheDir.entries()) {
      if (handler.kind === 'file' && !name.startsWith(PREFIX_METADATA)) {
        result.push({
          name,
          size: await (handler as FileSystemFileHandle)
            .getFile()
            .then((f) => f.size),
          metadata: metadataMap[name] || {
            // try to polyfill for old versions
            originalSize: (await (handler as FileSystemFileHandle).getFile())
              .size,
            originalURL: '',
            etag: '',
          },
        });
      }
    }
    return result;
  }

  /**
   * Clear all files currently in cache
   */
  async clear(): Promise<void> {
    await this.deleteMany(() => true);
  }

  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  async delete(nameOrURL: string): Promise<void> {
    const name2 = await this.getNameFromURL(nameOrURL);
    await this.deleteMany(
      (entry) => entry.name === nameOrURL || entry.name === name2
    );
  }

  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  async deleteMany(predicate: (e: CacheEntry) => boolean): Promise<void> {
    const cacheDir = await getCacheDir();
    const list = await this.list();
    for (const item of list) {
      if (predicate(item)) {
        cacheDir.removeEntry(item.name);
      }
    }
  }

  /**
   * Write the metadata of the file to disk.
   *
   * This function is separated from `write()` for compatibility reason. In older version of wllama, there was no metadata for cached file, so when newer version of wllama loads a file created by older version, it will try to polyfill the metadata.
   */
  async writeMetadata(
    name: string,
    metadata: CacheEntryMetadata
  ): Promise<void> {
    const blob = new Blob([JSON.stringify(metadata)], { type: 'text/plain' });
    await opfsWrite(name, blob.stream(), PREFIX_METADATA);
  }
}

export default CacheManager;

/**
 * Write to OPFS file from ReadableStream
 */
async function opfsWrite(
  key: string,
  stream: ReadableStream,
  prefix = ''
): Promise<void> {
  try {
    const fileName = await urlToFileName(key, prefix);
    const writable = await opfsWriteViaWorker(fileName);
    await writable.truncate(0); // clear file content
    const reader = stream.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await writable.write(value);
    }
    await writable.close();
  } catch (e) {
    console.error('opfsWrite', e);
  }
}

/**
 * Opens a file in OPFS for reading
 * @returns ReadableStream
 */
async function opfsOpen(
  originalURLOrName: string,
  prefix = ''
): Promise<File | null> {
  const getFileHandler = async (fname: string) => {
    try {
      const cacheDir = await getCacheDir();
      const fileHandler = await cacheDir.getFileHandle(fname);
      return await fileHandler.getFile();
    } catch (e) {
      // TODO: check if exception is NotFoundError
      return null;
    }
  };
  let handler = await getFileHandler(originalURLOrName);
  if (handler) {
    return handler;
  }
  // retry if needed
  const fileName = await urlToFileName(originalURLOrName, prefix);
  handler = await getFileHandler(fileName);
  return handler;
}

/**
 * Get file size of a file in OPFS
 * @returns number of bytes, or -1 if file does not exist
 */
async function opfsFileSize(originalURL: string, prefix = ''): Promise<number> {
  try {
    const cacheDir = await getCacheDir();
    const fileName = await urlToFileName(originalURL, prefix);
    const fileHandler = await cacheDir.getFileHandle(fileName);
    const file = await fileHandler.getFile();
    return file.size;
  } catch (e) {
    // TODO: check if exception is NotFoundError
    return -1;
  }
}

async function urlToFileName(url: string, prefix: string) {
  const hashBuffer = await crypto.subtle.digest(
    'SHA-1',
    new TextEncoder().encode(url)
  );
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  return `${prefix}${hashHex}_${url.split('/').pop()}`;
}

async function getCacheDir() {
  const opfsRoot = await navigator.storage.getDirectory();
  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });
  return cacheDir;
}

async function opfsWriteViaWorker(fileName: string): Promise<{
  truncate(offset: number): Promise<void>;
  write(value: Uint8Array): Promise<void>;
  close(): Promise<void>;
}> {
  const worker = createWorker(OPFS_UTILS_WORKER_CODE);
  let pResolve: (v: any) => void;
  let pReject: (v: any) => void;
  worker.onmessage = (e: MessageEvent<any>) => {
    if (e.data.ok) pResolve(null);
    else if (e.data.err) pReject(e.data.err);
  };
  const workerExec = (data: {
    open?: string;
    value?: Uint8Array;
    done?: boolean;
  }) =>
    new Promise<void>((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      // TODO @ngxson : Safari mobile doesn't support transferable ArrayBuffer
      worker.postMessage(
        data,
        isSafariMobile()
          ? undefined
          : {
              transfer: data.value ? [data.value.buffer] : [],
            }
      );
    });
  await workerExec({ open: fileName });
  return {
    truncate: async () => {
      /* noop */
    },
    write: (value) => workerExec({ value }),
    close: async () => {
      await workerExec({ done: true });
      worker.terminate();
    },
  };
}
