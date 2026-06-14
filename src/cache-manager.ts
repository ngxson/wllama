import { getHFFileSHA256 } from './huggingface';
import type { DownloadProgressCallback } from './model-manager';
import { COSBackend } from './storage/cos';
import type { StorageBackend, StorageFileHint } from './storage/index';

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
  /**
   * Optional SHA256, mostly used by COS backend
   */
  sha256?: string | undefined;
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

    if (hint && (await this.sb.getSize(fileKey, hint)) !== -1) {
      // File already in COS; metadata was written on the original download.
      return;
    }

    const response = await fetch(url, {
      ...(options.headers ? { headers: options.headers } : {}),
      ...(options.signal ? { signal: options.signal } : {}),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
    }

    const contentLength = response.headers.get('content-length');
    const etag = (response.headers.get('etag') || '').replace(
      /[^A-Za-z0-9]/g,
      ''
    );
    const total = parseInt(contentLength ?? '0', 10);

    const progressCallback = options.progressCallback;
    let loaded = 0;
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

    const metadata: CacheEntryMetadata = {
      originalURL: url,
      originalSize: total,
      etag,
      ...(options.metadataAdditional ?? {}),
    };
    if (sha256) {
      metadata.sha256 = sha256;
    }

    await this.sb.write(
      fileKey,
      response.body.pipeThrough(progressStream),
      hint
    );
    await this.writeMetadata(fileKey, metadata);
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
    const blob = await this.sb.read(`${PREFIX_METADATA}${name}`);
    const cachedSize = await this.sb.getSize(name);
    if (!blob) {
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
