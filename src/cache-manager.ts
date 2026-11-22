import { getHFFileSHA256 } from './huggingface';
import type { DownloadProgressCallback } from './model-manager';
import { COSBackend } from './storage/cos';
import type { StorageBackend, StorageFileHint } from './storage/index';
import { delay } from './utils';

const PREFIX_METADATA = '__metadata__';
// metadata of a partial file, stored as `${PREFIX_METADATA}${PREFIX_PARTIAL}${name}`
const PREFIX_PARTIAL = '__partial__';
const DEFAULT_RETRIES = 5;
const DEFAULT_CONNECTIONS = 4;
export const DOWNLOAD_CONFIG = {
  partsMinSize: 64 * 1024 * 1024, // smaller files use one connection
  partSize: 32 * 1024 * 1024,
  savePrefixEvery: 256 * 1024 * 1024, // how often to save resume progress
};

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
  /**
   * If the connection drops, the download continues from the last byte (when the server and the storage support it).
   * Number of tries in a row without new bytes before giving up. Default: 5, set to 0 to fail at once
   */
  retries?: number;
  /**
   * Connections per file (default: 4), for files of 64MB or more
   */
  connections?: number;
};

// To prevent breaking change, we fill etag with a pre-defined value
export const POLYFILL_ETAG = 'polyfill_for_older_version';

export interface CacheEntry {
  /**
   * Storage key for this file, in the format: `${hashSHA1(fullURL)}_${fileName}`
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
  draftURL?: string | undefined;
  loraURLs?: string[] | undefined;
  /**
   * Optional SHA256, mostly used by COS backend
   */
  sha256?: string | undefined;
  /**
   * Download in parts: size of the complete start of the file
   */
  contiguous?: number | undefined;
}

function hintFromMetadata(
  metadata: CacheEntryMetadata | null
): StorageFileHint | undefined {
  if (!metadata) return undefined;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return undefined;
}

/**
 * Manages cached model files, backed by a pluggable StorageBackend.
 *
 * Defaults to OPFS (Origin Private File System).
 */
export class CacheManager {
  private sb: StorageBackend;

  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends: StorageBackend[] = [new COSBackend()]) {
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error('No supported storage backend found');
  }

  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  async getNameFromURL(url: string): Promise<string> {
    return urlToFileName(url, '');
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
    // write file first, then metadata
    await this.sb.write(name, stream);
    await this.writeMetadata(name, metadata);
  }

  async download(url: string, options: DownloadOptions = {}): Promise<void> {
    const fileKey = await urlToFileName(url, '');

    // Fetch sha256 before the GET so we can skip the download entirely if the
    // file is already in COS (avoids opening a connection just to cancel it).
    const sha256 = await getHFFileSHA256(url, options.headers ?? {});
    const hint = sha256 ? { sha256 } : undefined;
    const canResume = this.sb.canWriteAt?.(hint) ?? false;

    // a partial file left by an interrupted download, continue it from here
    let offset = 0;
    let etag = ''; // exact ETag of the bytes on disk

    const cachedSize = await this.sb.getSize(fileKey, hint);
    if (cachedSize !== -1) {
      const metadata = await this.readMetadata(fileKey);
      if (
        metadata?.originalURL === url &&
        metadata.originalSize === cachedSize
      ) {
        // the file is reused by another model: its mmproj, draft and adapters change
        const updated = { ...metadata, ...(options.metadataAdditional ?? {}) };
        if (JSON.stringify(updated) !== JSON.stringify(metadata)) {
          await this.writeMetadata(fileKey, updated);
        }
        return;
      }
      const partial = metadata
        ? null
        : await this.readMetadata(`${PREFIX_PARTIAL}${fileKey}`);
      // parts are written out of order, the file can have holes
      const prefix = Math.min(partial?.contiguous ?? cachedSize, cachedSize);
      if (
        canResume &&
        partial?.originalURL === url &&
        isStrongEtag(partial.etag) &&
        prefix < partial.originalSize
      ) {
        offset = prefix;
        etag = partial.etag;
      } else {
        // no metadata, file maybe partial; ask server for the real size
        const head = await fetch(url, {
          method: 'HEAD',
          ...(options.headers ? { headers: options.headers } : {}),
        });
        const originalSize = parseInt(
          head.headers.get('content-length') ?? '0',
          10
        );
        const headEtag = cleanEtag(head.headers.get('etag'));
        const complete =
          partial?.contiguous === undefined ||
          partial.contiguous === originalSize;
        if (originalSize > 0 && originalSize === cachedSize && complete) {
          await this.writeMetadata(fileKey, {
            originalURL: url,
            originalSize,
            etag: headEtag,
            sha256,
            ...(options.metadataAdditional ?? {}),
          });
          return;
        }
        // size mismatch, drop the file and download it again
        await this.sb.delete(fileKey);
        await this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
    }

    const maxRetries = options.retries ?? DEFAULT_RETRIES;
    const progressCallback = options.progressCallback;
    let total = 0;
    let loaded = 0;
    let failures = 0; // failures in a row without new bytes
    // largest part of the file kept on disk so far, a try that does not grow it is a failure
    // it never goes down, so the retries always run out
    let kept = offset;
    // true if the bytes on disk can be kept for the next try
    let resumable = offset > 0;
    // set by downloadParts(), kept until a stream write truncates the file
    let partsPrefix: number | null = null;

    while (true) {
      try {
        const response = await fetch(url, {
          headers: {
            ...(options.headers ?? {}),
            // a 206 means ranges are supported
            ...(offset > 0 || canResume ? { Range: `bytes=${offset}-` } : {}),
          },
          ...(options.signal ? { signal: options.signal } : {}),
        });

        if (response.status === 416 && offset > 0) {
          // the file on the server is smaller than the partial file
          offset = 0;
          resumable = false;
          continue;
        }
        if (!response.ok || !response.body) {
          throw new DownloadHttpError(
            `Failed to fetch ${url}: HTTP ${response.status}`
          );
        }

        const resEtag = response.headers.get('etag') ?? '';
        const range = parseContentRange(response);
        if (offset > 0) {
          const sameFile =
            response.status === 206 &&
            resEtag === etag &&
            range?.start === offset;
          if (!sameFile) {
            // the file changed on the server, or the server ignored the range
            offset = 0;
            resumable = false;
            if (response.status === 206) {
              await response.body.cancel();
              continue;
            }
          }
        }
        etag = resEtag;
        // on a 206, content-length is only the size of this part
        total =
          response.status === 206
            ? (range?.total ?? 0)
            : parseInt(response.headers.get('content-length') ?? '0', 10);

        // a weak ETag can be the same for different bytes, do not resume with it
        resumable = canResume && isStrongEtag(etag) && total > 0;
        if (resumable && offset === 0) {
          // drop the old bytes first, so this metadata never goes with the start of another version
          await this.sb.delete(fileKey);
          // written before the body, so a later call can resume this file
          await this.writeMetadata(`${PREFIX_PARTIAL}${fileKey}`, {
            originalURL: url,
            originalSize: total,
            etag,
          });
        }

        loaded = offset;
        const connections = options.connections ?? DEFAULT_CONNECTIONS;
        if (
          resumable &&
          response.status === 206 &&
          connections > 1 &&
          total - offset >= DOWNLOAD_CONFIG.partsMinSize &&
          this.sb.openWriter
        ) {
          let lastAt = 0;
          await this.downloadParts({
            url,
            fileKey,
            hint,
            response,
            offset,
            total,
            etag,
            connections,
            options,
            onBytes: (n) => {
              loaded += n;
              const now = Date.now();
              if (progressCallback && now - lastAt > 100) {
                lastAt = now;
                progressCallback({ loaded, total });
              }
            },
            onPrefix: (prefix) => (partsPrefix = prefix),
          });
          progressCallback?.({ loaded, total });
          if (loaded < total) {
            throw new Error(`Incomplete download: ${loaded} of ${total} bytes`);
          }
          break;
        }
        let lastProgressAt = 0;
        const progressStream = new TransformStream<Uint8Array, Uint8Array>({
          transform(chunk, controller) {
            loaded += chunk.byteLength;
            if (progressCallback) {
              const now = Date.now();
              if (now - lastProgressAt > 100) {
                lastProgressAt = now;
                progressCallback({ loaded, total });
              }
            }
            controller.enqueue(chunk);
          },
          flush() {
            progressCallback?.({ loaded, total: total || loaded });
          },
        });

        partsPrefix = null;
        await this.sb.write(
          fileKey,
          response.body.pipeThrough(progressStream),
          hint,
          offset
        );
        // the server can send less than the requested range
        if (total > 0 && loaded < total) {
          throw new Error(`Incomplete download: ${loaded} of ${total} bytes`);
        }
        break;
      } catch (e: any) {
        const fatal =
          e instanceof DownloadHttpError ||
          options.signal?.aborted ||
          e?.name === 'QuotaExceededError' ||
          // another download has the same file open
          e?.name === 'NoModificationAllowedError';
        if (fatal) throw e;
        // keep the bytes on disk only if the next try can check that the file did not change
        const size = resumable
          ? (partsPrefix ?? (await this.sb.getSize(fileKey, hint)))
          : 0;
        if (size > kept) {
          failures = 0;
          kept = size;
        } else {
          failures++;
        }
        if (maxRetries === 0 || failures > maxRetries) throw e;
        offset = Math.max(size, 0);
        if (failures > 0) {
          await delay(1000 * 2 ** (failures - 1));
          options.signal?.throwIfAborted();
        }
      }
    }

    const metadata: CacheEntryMetadata = {
      originalURL: url,
      // without Content-Length, the size is only known once the body is fully read
      originalSize: total || loaded,
      etag: cleanEtag(etag),
      ...(options.metadataAdditional ?? {}),
    };
    if (sha256) {
      metadata.sha256 = sha256;
    }
    await this.writeMetadata(fileKey, metadata);
    await this.sb.delete(`${PREFIX_METADATA}${PREFIX_PARTIAL}${fileKey}`);
  }

  // part 0 comes from `response`, the others from `connections` workers
  private async downloadParts(p: {
    url: string;
    fileKey: string;
    hint: StorageFileHint | undefined;
    response: Response;
    offset: number;
    total: number;
    etag: string;
    connections: number;
    options: DownloadOptions;
    onBytes: (n: number) => void;
    onPrefix: (prefix: number) => void;
  }): Promise<void> {
    const { url, fileKey, offset, total, etag } = p;
    const PART_SIZE = DOWNLOAD_CONFIG.partSize;
    const nParts = Math.ceil((total - offset) / PART_SIZE);
    const partRange = (i: number) =>
      [
        offset + i * PART_SIZE,
        Math.min(total, offset + (i + 1) * PART_SIZE),
      ] as const;

    // one at a time: two writes of the same OPFS file fail with NoModificationAllowedError
    let saving = Promise.resolve();
    const saveMetadata = (contiguous: number) =>
      (saving = saving.then(() =>
        this.writeMetadata(`${PREFIX_PARTIAL}${fileKey}`, {
          originalURL: url,
          originalSize: total,
          etag,
          contiguous,
        })
      ));
    await saveMetadata(offset);
    p.onPrefix(offset);

    const writer = await this.sb.openWriter!(fileKey, offset, p.hint);
    const abort = new AbortController();
    const onAbort = () => abort.abort();
    p.options.signal?.addEventListener('abort', onAbort);

    const done = new Array<boolean>(nParts).fill(false);
    let firstNotDone = 0;
    let saved = offset;
    const finish = async (i: number) => {
      done[i] = true;
      while (firstNotDone < nParts && done[firstNotDone]) firstNotDone++;
      const prefix = Math.min(total, offset + firstNotDone * PART_SIZE);
      p.onPrefix(prefix);
      if (prefix - saved >= DOWNLOAD_CONFIG.savePrefixEvery) {
        saved = prefix;
        await saveMetadata(prefix);
      }
    };
    const pump = async (
      body: ReadableStream<Uint8Array>,
      start: number,
      end: number
    ) => {
      const reader = body.getReader();
      // the first response is not fetched with abort.signal
      const onAbort = () => reader.cancel().catch(() => {});
      abort.signal.addEventListener('abort', onAbort);
      let pos = start;
      try {
        while (pos < end) {
          const { done, value } = await reader.read();
          if (done && abort.signal.aborted) {
            throw new DOMException('Aborted', 'AbortError');
          }
          if (done) throw new Error(`Incomplete part: ${pos} of ${end} bytes`);
          const n = Math.min(value.byteLength, end - pos);
          // write() transfers the buffer
          await writer.write(pos, value.subarray(0, n));
          pos += n;
          p.onBytes(n);
        }
      } finally {
        abort.signal.removeEventListener('abort', onAbort);
        reader.cancel().catch(() => {});
      }
    };
    let next = 1;
    const worker = async () => {
      while (next < nParts) {
        const i = next++;
        const [start, end] = partRange(i);
        const res = await fetch(url, {
          headers: {
            ...(p.options.headers ?? {}),
            Range: `bytes=${start}-${end - 1}`,
          },
          signal: abort.signal,
        });
        if (
          res.status !== 206 ||
          !res.body ||
          (res.headers.get('etag') ?? '') !== etag
        ) {
          res.body?.cancel().catch(() => {});
          throw new Error(
            `The file changed on the server or the range is not supported: ${url}`
          );
        }
        await pump(res.body, start, end);
        await finish(i);
      }
    };
    const first = async () => {
      await pump(p.response.body!, ...partRange(0));
      await finish(0);
      await worker();
    };

    // the first error stops the other connections
    const tasks = [
      first(),
      ...Array.from({ length: p.connections - 1 }, worker),
    ].map((t) =>
      t.catch((e) => {
        abort.abort();
        throw e;
      })
    );
    const results = await Promise.allSettled(tasks);
    p.options.signal?.removeEventListener('abort', onAbort);
    await writer.close();
    const errors = results.flatMap((r) =>
      r.status === 'rejected' ? [r.reason] : []
    );
    if (errors.length > 0) {
      if (p.options.signal?.aborted) throw p.options.signal.reason;
      // skip the AbortErrors of the stopped connections
      throw errors.find((e) => e?.name !== 'AbortError') ?? errors[0];
    }
  }

  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  async open(nameOrURL: string): Promise<Blob | null> {
    const hint1 = hintFromMetadata(await this.getMetadata(nameOrURL));
    const direct = await this.sb.read(nameOrURL, hint1);
    if (direct) return direct;
    // also accept the original URL
    const key = await urlToFileName(nameOrURL, '');
    const hint2 = hintFromMetadata(await this.getMetadata(key));
    return this.sb.read(key, hint2);
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
    const hint = hintFromMetadata(await this.getMetadata(name));
    return this.sb.getSize(name, hint);
  }

  /**
   * Get metadata of a cached file
   */
  async getMetadata(name: string): Promise<CacheEntryMetadata | null> {
    const metadata = await this.readMetadata(name);
    if (metadata) return metadata;
    const cachedSize = await this.sb.getSize(name);
    return cachedSize > 0
      ? // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: '',
        }
      : // cached file not found
        null;
  }

  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  private async readMetadata(name: string): Promise<CacheEntryMetadata | null> {
    const blob = await this.sb.read(`${PREFIX_METADATA}${name}`);
    if (!blob) return null;
    try {
      return await new Response(blob).json();
    } catch (e) {
      // metadata corrupted; caller will re-download
      return null;
    }
  }

  /**
   * List all files currently in cache
   */
  async list(): Promise<CacheEntry[]> {
    const all = await this.sb.list();
    const metadataMap: Record<string, CacheEntryMetadata> = {};

    for (const { key } of all) {
      if (key.startsWith(PREFIX_METADATA)) {
        const blob = await this.sb.read(key);
        if (blob) {
          const meta = await new Response(blob).json().catch(() => null);
          metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
        }
      }
    }

    const result: CacheEntry[] = [];
    for (const { key, size } of all) {
      if (!key.startsWith(PREFIX_METADATA)) {
        result.push({
          name: key,
          size,
          metadata: metadataMap[key] || {
            originalSize: size,
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
    const list = await this.list();
    for (const item of list) {
      if (predicate(item)) {
        await this.sb.delete(item.name);
        await this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        await this.sb.delete(`${PREFIX_METADATA}${PREFIX_PARTIAL}${item.name}`);
      }
    }
  }

  /**
   * Write the metadata of the file to disk.
   */
  async writeMetadata(
    name: string,
    metadata: CacheEntryMetadata
  ): Promise<void> {
    const blob = new Blob([JSON.stringify(metadata)], { type: 'text/plain' });
    await this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
  }
}

export default CacheManager;

class DownloadHttpError extends Error {}

function cleanEtag(etag: string | null): string {
  return (etag || '').replace(/[^A-Za-z0-9]/g, '');
}

function isStrongEtag(etag: string | undefined): boolean {
  return !!etag && !etag.startsWith('W/');
}

// e.g. "bytes 1024-2047/4096" -> { start: 1024, total: 4096 }; total is 0 if unknown
function parseContentRange(
  response: Response
): { start: number; total: number } | null {
  const match = /^bytes (\d+)-\d+\/(\d+|\*)$/.exec(
    response.headers.get('content-range') ?? ''
  );
  if (!match) return null;
  return {
    start: parseInt(match[1], 10),
    total: match[2] === '*' ? 0 : parseInt(match[2], 10),
  };
}

async function urlToFileName(url: string, prefix: string): Promise<string> {
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
