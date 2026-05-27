import type { DownloadProgressCallback } from './model-manager';
import { createWorker, isSafariMobile } from './utils';
import { OPFS_UTILS_WORKER_CODE } from './workers-code/generated';
import {
  isCrossOriginStorageAvailable,
  resolveUrlHash,
  setUrlHash,
  tryReadFromCOS,
  writeToCosByHash,
} from './cross-origin-storage';

const PREFIX_METADATA = '__metadata__';

export type DownloadOptions = {
  /**
   * Callback function to track download progress
   */
  progressCallback?: DownloadProgressCallback;
  /**
   * Additional metadata to be stored with the downloaded file
   */
  metadataAdditional?: Record<string, any>;
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
  /**
   * URL to mmproj file, if exists
   */
  mmprojURL?: string | undefined;
  /**
   * SHA-256 content hash, used as the key for Cross-Origin Storage.
   * Present only when the file was downloaded or served via COS.
   */
  sha256?: string | undefined;
}

/**
 * Cache implementation using OPFS (Origin private file system)
 *
 * This class is also responsible for downloading files from the internet.
 */
export class CacheManager {
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
    if (isCrossOriginStorageAvailable() && !options.signal?.aborted) {
      const hash = await resolveUrlHash(url);
      if (hash) {
        const cosBlob = await tryReadFromCOS(hash);
        if (cosBlob) {
          // File already in COS – record metadata only.
          const metadata: CacheEntryMetadata = {
            etag: POLYFILL_ETAG,
            originalSize: cosBlob.size,
            originalURL: url,
            sha256: hash,
            ...(options.metadataAdditional ?? {}),
          };
          await this.writeMetadata(url, metadata);
          options.progressCallback?.({
            loaded: cosBlob.size,
            total: cosBlob.size,
          });
          return;
        }
      }

      // File not in COS (or hash unknown for non-HF URLs) — download directly
      // into COS, computing the SHA-256 on-the-fly when needed.  No OPFS data
      // is written on this path.
      try {
        await this.downloadToCOS(url, hash, options);
        return;
      } catch (e) {
        if (options.signal?.aborted) throw e;
        console.warn('[wllama] COS download failed, falling back to OPFS:', e);
        // Fall through to the OPFS worker path below.
      }
    }

    // OPFS fallback: COS unavailable or downloadToCOS failed.
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
        metadataAdditional: options.metadataAdditional ?? {},
        options: { headers: options.headers, aborted },
      });
      worker.onmessage = (e: MessageEvent<any>) => {
        if (e.data.ok) {
          worker.terminate();
          void this.writeToCosBg(url);
          resolve();
        } else if (e.data.err) {
          worker.terminate();
          reject(e.data.err);
        } else if (e.data.progress) {
          const progress: { loaded: number; total: number } = e.data.progress;
          options.progressCallback?.(progress);
        } else {
          reject(new Error('Unknown message from worker'));
          console.error('Unknown message from worker', e.data);
        }
      };
    });
  }

  /**
   * Download a file from the network directly into Cross-Origin Storage
   * without writing any data to OPFS.  On success only a tiny metadata JSON
   * is written to OPFS.
   *
   * When `hash` is supplied (HF LFS pointer was available) the chunks are
   * written directly to the COS writable one-by-one — no Blob assembly.
   * When `hash` is null (non-HF URL) the chunks are first concatenated into
   * a single buffer so SHA-256 can be computed, then that buffer is written.
   * Either way the data never touches OPFS.
   */
  private async downloadToCOS(
    url: string,
    hash: string | null,
    options: DownloadOptions
  ): Promise<void> {
    const response = await fetch(url, {
      headers: options.headers,
      signal: options.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error(`Network error: ${response.status} ${response.statusText}`);
    }

    const total = Number(response.headers.get('content-length') || '0');
    let loaded = 0;

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      options.progressCallback?.({ loaded, total });
    }

    // For unknown-hash URLs: assemble a contiguous buffer and compute SHA-256.
    let writeBlob: Blob;
    if (!hash) {
      const combined = new Uint8Array(loaded);
      let off = 0;
      for (const c of chunks) { combined.set(c, off); off += c.byteLength; }
      chunks.length = 0; // release individual chunks before hashing
      const hashBuf = await crypto.subtle.digest('SHA-256', combined.buffer);
      hash = Array.from(new Uint8Array(hashBuf))
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
      await setUrlHash(url, hash);
      writeBlob = new Blob([combined]);
    } else {
      // Blob constructor accepts Uint8Array[] as BlobPart[].
      writeBlob = new Blob(chunks as any[]);
    }

    const [handle] = await navigator.crossOriginStorage.requestFileHandles(
      [{ algorithm: 'SHA-256', value: hash }],
      { create: true }
    );
    const writable = await (handle as any).createWritable();
    await writable.write(writeBlob);
    await writable.close();

    const metadata: CacheEntryMetadata = {
      etag: POLYFILL_ETAG,
      originalSize: loaded,
      originalURL: url,
      sha256: hash,
      ...(options.metadataAdditional ?? {}),
    };
    await this.writeMetadata(url, metadata);
  }

  /**
   * Background task: write a file from OPFS to Cross-Origin Storage after a
   * successful network download.  Errors are swallowed so a COS failure never
   * surfaces to the caller.
   *
   * Hash strategy:
   * 1. Use the hash already resolved by `resolveUrlHash` (e.g. from the LFS
   *    pointer fetch that happened in the COS pre-check above).
   * 2. For non-HF URLs where no LFS hash exists, skip COS – computing SHA-256
   *    of a multi-GB model file in the browser main thread is impractical.
   */
  private async writeToCosBg(url: string): Promise<void> {
    if (!isCrossOriginStorageAvailable()) return;
    try {
      const hash = await resolveUrlHash(url);
      if (!hash) return;

      const filename = await urlToFileName(url, '');
      const blob = await opfsOpen(filename);
      if (!blob) return;

      await writeToCosByHash(blob, hash);

      // COS write succeeded — update metadata with sha256 and persist the hash.
      // We intentionally do NOT remove the OPFS data file here: the Worker may
      // already hold a File handle to it and a concurrent removeEntry() would
      // cause a NotFoundError mid-read.  The OPFS data will be cleaned up on
      // the next clear() or deleteMany() call.
      const meta = await this.getMetadata(url);
      if (meta) {
        await this.writeMetadata(url, { ...meta, sha256: hash });
      }
      await setUrlHash(url, hash);
    } catch {
      // Non-fatal – if COS write or cleanup fails, the file stays in OPFS.
    }
  }

  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  async open(nameOrURL: string): Promise<Blob | null> {
    const fromOpfs = await opfsOpen(nameOrURL);
    if (fromOpfs) return fromOpfs;
    // COS fallback: if the data file is absent but the metadata records a
    // sha256, serve the blob directly from Cross-Origin Storage.
    // The metadata file is stored as PREFIX_METADATA + OPFS data filename.
    if (isCrossOriginStorageAvailable()) {
      const metaFile = await opfsOpen(PREFIX_METADATA + nameOrURL);
      if (metaFile) {
        try {
          const meta: CacheEntryMetadata = await new Response(
            metaFile.stream()
          ).json();
          if (meta.sha256) {
            const cosBlob = await tryReadFromCOS(meta.sha256);
            if (cosBlob) return cosBlob;
            // COS resource was evicted — remove stale metadata so the next
            // validate() returns INVALID and triggers a fresh download.
            try {
              const cacheDir = await getCacheDir();
              await cacheDir.removeEntry(PREFIX_METADATA + nameOrURL);
            } catch {}
          }
        } catch {}
      }
    }
    return null;
  }

  /**
   * Check whether a file is available in Cross-Origin Storage without downloading it.
   *
   * Resolves the SHA-256 hash for the URL (via the HF Git LFS pointer file or a
   * previously-cached hash entry) and queries `navigator.crossOriginStorage`.
   *
   * Returns the `Blob` if found in COS, `null` if the API is unavailable, the
   * file is not cached cross-origin, or the hash cannot be determined.
   *
   * Callers can use the returned size to calculate total download size without
   * making a HEAD request to the origin server.
   */
  async checkCOS(url: string): Promise<Blob | null> {
    if (!isCrossOriginStorageAvailable()) return null;
    const hash = await resolveUrlHash(url);
    if (!hash) return null;
    return tryReadFromCOS(hash);
  }

  /**
   * Resolve a WASM URL through Cross-Origin Storage, avoiding a network
   * round-trip when the binary is already cached cross-origin.
   *
   * Strategy:
   * 1. If COS is unavailable, return the original URL unchanged.
   * 2. If the SHA-256 hash is already in the hash cache (from a previous
   *    load), attempt a COS lookup. On hit, return a `blob:` object URL
   *    so the Worker never touches the network.
   * 3. On COS miss, or when no hash is cached yet, fetch the WASM
   *    normally, compute its SHA-256, write it to COS in the background,
   *    and return a `blob:` object URL so the caller skips a second fetch.
   * 4. On any error, return the original URL as a safe fallback.
   *
   * WASM files have no HF LFS pointer, so the hash is computed from
   * content on first load and then persisted in the hash cache. WASM
   * files are small enough (a few MB) that buffering them is fine.
   *
   * Note: the returned `blob:` URL is never explicitly revoked; it lives
   * until the page is unloaded, consistent with other blob URLs in wllama.
   */
  async resolveWasmUrl(wasmUrl: string): Promise<string> {
    if (!isCrossOriginStorageAvailable()) return wasmUrl;

    // COS path is non-fatal: any failure falls through to the network fetch.
    let hash: string | null = null;
    try {
      hash = await resolveUrlHash(wasmUrl);
      if (hash) {
        const cosBlob = await tryReadFromCOS(hash);
        if (cosBlob) {
          // Re-wrap to guarantee the correct MIME type for
          // WebAssembly.instantiateStreaming() in the Worker.
          const blob =
            cosBlob.type === 'application/wasm'
              ? cosBlob
              : new Blob([cosBlob], { type: 'application/wasm' });
          return URL.createObjectURL(blob);
        }
      }
    } catch {
      hash = null;
    }

    // COS miss — fetch from the network.
    // This is intentionally outside the try/catch: a 404 or network error
    // here means the WASM is genuinely unavailable.  Returning wasmUrl as a
    // fallback would only cause the Worker to make the same failing request
    // and crash with a cryptic "expected magic word" error.
    const response = await fetch(wasmUrl);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch WASM (${response.status} ${response.statusText}): ${wasmUrl}`
      );
    }
    const buf = await response.arrayBuffer();

    if (!hash) {
      // Compute and persist the hash so future loads can skip this fetch.
      const hashBuf = await crypto.subtle.digest('SHA-256', buf);
      hash = Array.from(new Uint8Array(hashBuf))
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
      await setUrlHash(wasmUrl, hash);
    }

    // Write to COS in the background; return a blob: URL from the
    // already-fetched buffer so the Worker doesn't fetch it again.
    const blob = new Blob([buf], { type: 'application/wasm' });
    void writeToCosByHash(blob, hash);
    return URL.createObjectURL(blob);
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
    const metadataMap: Record<string, CacheEntryMetadata> = {};
    const dataSizeMap: Record<string, number> = {};
    // @ts-ignore
    for await (const [name, handler] of cacheDir.entries()) {
      if (handler.kind !== 'file') continue;
      const file = await (handler as FileSystemFileHandle).getFile();
      if (name.startsWith(PREFIX_METADATA)) {
        const meta = await new Response(file.stream()).json().catch(() => null);
        if (meta) metadataMap[name.slice(PREFIX_METADATA.length)] = meta;
      } else {
        dataSizeMap[name] = file.size;
      }
    }
    const result: CacheEntry[] = [];
    const allNames = new Set([
      ...Object.keys(metadataMap),
      ...Object.keys(dataSizeMap),
    ]);
    for (const name of allNames) {
      const meta = metadataMap[name];
      const opfsSize = dataSizeMap[name];
      result.push({
        name,
        // COS-backed entries have no OPFS data file; use the recorded size.
        size: opfsSize ?? meta?.originalSize ?? 0,
        metadata: meta ?? {
          etag: POLYFILL_ETAG,
          originalSize: opfsSize ?? 0,
          originalURL: '',
        },
      });
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
        // Data file may be absent for COS-backed entries; swallow the error.
        try {
          await cacheDir.removeEntry(item.name);
        } catch {}
        try {
          await cacheDir.removeEntry(PREFIX_METADATA + item.name);
        } catch {}
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
  const workerExec = (
    data: Record<string, unknown>,
    transferBuf?: ArrayBuffer
  ) =>
    new Promise<void>((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      // TODO @ngxson : Safari mobile doesn't support transferable ArrayBuffer
      worker.postMessage(
        data,
        isSafariMobile() || !transferBuf
          ? undefined
          : { transfer: [transferBuf] }
      );
    });
  await workerExec({ action: 'open', filename: fileName });
  return {
    truncate: async () => {
      /* noop */
    },
    write: (value) => workerExec({ action: 'write', buf: value }, value.buffer),
    close: async () => {
      await workerExec({ action: 'close' });
      worker.terminate();
    },
  };
}
