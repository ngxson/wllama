// Adapted from https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/utils/WebBlob.ts

import CacheManager, {
  CacheEntryMetadata,
  POLYFILL_ETAG,
} from '../cache-manager';

type ProgressCallback = (opts: { loaded: number; total: number }) => any;

/**
 * WebBlob is a Blob implementation for web resources that supports range requests.
 */

interface GGUFRemoteBlobCreateOptions {
  /**
   * Custom fetch function to use instead of the default one, for example to use a proxy or edit headers.
   */
  fetch?: typeof fetch;
  useCache?: boolean;
  progressCallback?: ProgressCallback;
  startSignal?: Promise<void>;
  allowOffline: boolean;
  cacheManager: CacheManager;
  /**
   * Should we skip TEE the output stream?
   * Set to true if we only download model to cache, without reading it
   */
  noTEE: boolean;
  /**
   * Custom debug logger
   */
  logger?: {
    debug: (typeof console)['debug'];
  };
}

export class GGUFRemoteBlob extends Blob {
  static async create(
    url: string,
    opts: GGUFRemoteBlobCreateOptions
  ): Promise<Blob> {
    const { cacheManager } = opts;
    const customFetch = opts?.fetch ?? fetch;
    const cacheKey = url;
    let remoteFile: CacheEntryMetadata;

    try {
      const response = await customFetch(url, { method: 'HEAD' });
      remoteFile = {
        originalURL: url,
        originalSize: Number(response.headers.get('content-length')),
        etag: (response.headers.get('etag') || '').replace(/[^A-Za-z0-9]/g, ''),
        // supportRange: response.headers.get('accept-ranges') === 'bytes';
      };
    } catch (err) {
      // connection error (i.e. offline)
      if (opts.allowOffline) {
        const cachedMeta = await cacheManager.getMetadata(cacheKey);
        if (cachedMeta) {
          remoteFile = cachedMeta;
        } else {
          throw new Error(
            'Network error, cannot find requested model in cache for using offline'
          );
        }
      } else {
        throw err;
      }
    }

    const cachedFileSize = await cacheManager.getSize(cacheKey);
    const cachedFile = await cacheManager.getMetadata(cacheKey);
    const skipCache = opts?.useCache === false;

    // migrate from old version: if metadata is polyfilled, we save the new metadata
    const metadataPolyfilled = cachedFile?.etag === POLYFILL_ETAG;
    if (metadataPolyfilled) {
      await cacheManager.writeMetadata(cacheKey, remoteFile);
    }

    const cachedFileValid =
      metadataPolyfilled ||
      (cachedFile &&
        remoteFile.etag === cachedFile.etag &&
        remoteFile.originalSize === cachedFileSize);
    if (cachedFileValid && !skipCache) {
      opts?.logger?.debug(`Using cached file ${cacheKey}`);
      const cachedFile = await cacheManager.open(cacheKey);
      (opts?.startSignal ?? Promise.resolve()).then(() => {
        opts?.progressCallback?.({
          loaded: cachedFileSize,
          total: cachedFileSize,
        });
      });
      return new GGUFRemoteBlob(
        url,
        0,
        remoteFile.originalSize,
        true,
        customFetch,
        {
          cachedStream: cachedFile!,
          progressCallback: () => {}, // unused
          etag: remoteFile.etag,
          noTEE: opts.noTEE,
          cacheManager: cacheManager,
        }
      );
    } else {
      if (remoteFile.originalSize !== cachedFileSize) {
        opts?.logger?.debug(
          `Cache file is present, but size mismatch (cache = ${cachedFileSize} bytes, remote = ${remoteFile.originalSize} bytes)`
        );
      }
      if (cachedFile && remoteFile.etag !== cachedFile.etag) {
        opts?.logger?.debug(
          `Cache file is present, but ETag mismatch (cache = "${cachedFile.etag}", remote = "${remoteFile.etag}")`
        );
      }
      opts?.logger?.debug(`NOT using cache for ${cacheKey}`);
      return new GGUFRemoteBlob(
        url,
        0,
        remoteFile.originalSize,
        true,
        customFetch,
        {
          progressCallback: opts?.progressCallback ?? (() => {}),
          startSignal: opts?.startSignal,
          etag: remoteFile.etag,
          noTEE: opts.noTEE,
          cacheManager: cacheManager,
        }
      );
    }
  }

  private cacheManager: CacheManager;
  private url: string;
  private etag: string;
  private start: number;
  private end: number;
  private contentType: string = '';
  private full: boolean;
  private fetch: typeof fetch;
  private cachedStream?: ReadableStream;
  private progressCallback: ProgressCallback;
  private startSignal?: Promise<void>;
  private noTEE: boolean;

  constructor(
    url: string,
    start: number,
    end: number,
    full: boolean,
    customFetch: typeof fetch,
    additionals: {
      cachedStream?: ReadableStream;
      progressCallback: ProgressCallback;
      startSignal?: Promise<void>;
      etag: string;
      noTEE: boolean;
      cacheManager: CacheManager;
    }
  ) {
    super([]);

    if (start !== 0) {
      throw new Error('start range must be 0');
    }

    this.url = url;
    this.start = start;
    this.end = end;
    this.contentType = '';
    this.full = full;
    this.fetch = customFetch;
    this.cachedStream = additionals.cachedStream;
    this.progressCallback = additionals.progressCallback;
    this.startSignal = additionals.startSignal;
    this.etag = additionals.etag;
    this.noTEE = additionals.noTEE;
    this.cacheManager = additionals.cacheManager;
  }

  override get size(): number {
    return this.end - this.start;
  }

  override get type(): string {
    return this.contentType;
  }

  override slice(): GGUFRemoteBlob {
    throw new Error('Unsupported operation');
  }

  override async arrayBuffer(): Promise<ArrayBuffer> {
    throw new Error('Unsupported operation');
  }

  override async text(): Promise<string> {
    throw new Error('Unsupported operation');
  }

  override stream(): ReturnType<Blob['stream']> {
    if (this.cachedStream) {
      return this.cachedStream;
    }

    const self = this;
    let loaded = 0;
    const stream = new TransformStream({
      transform(chunk, controller) {
        // if noTEE is set, we discard the chunk
        if (!self.noTEE) {
          controller.enqueue(chunk);
        }
        loaded += chunk.byteLength;
        self.progressCallback({
          loaded,
          total: self.size,
        });
      },
      // @ts-ignore unused variable
      flush(controller) {
        self.progressCallback({
          loaded: self.size,
          total: self.size,
        });
      },
    });

    (async () => {
      if (this.startSignal) {
        await this.startSignal;
      }
      this.fetchRange()
        .then((response) => {
          const [src0, src1] = response.body!.tee();
          src0.pipeThrough(stream);
          this.cacheManager.write(this.url, src1, {
            originalSize: this.end,
            originalURL: this.url,
            etag: this.etag,
          });
        })
        .catch((error) => stream.writable.abort(error.message));
    })();

    return stream.readable;
  }

  private fetchRange(): Promise<Response> {
    const fetch = this.fetch; // to avoid this.fetch() which is bound to the instance instead of globalThis
    if (this.full) {
      return fetch(this.url);
    }
    return fetch(this.url, {
      headers: {
        Range: `bytes=${this.start}-${this.end - 1}`,
      },
    });
  }
}
