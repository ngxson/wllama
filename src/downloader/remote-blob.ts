// Adapted from https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/utils/WebBlob.ts

import { CacheManager } from '../cache-manager';

type ProgressCallback = (opts: { loaded: number, total: number }) => any;

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
  /**
   * Custom debug logger
   */
  logger?: {
    debug: typeof console['debug'],
  };
}

export class GGUFRemoteBlob extends Blob {
  static async create(url: RequestInfo | URL, opts?: GGUFRemoteBlobCreateOptions): Promise<Blob> {
    const customFetch = opts?.fetch ?? fetch;

    const cacheKey = url.toString();
    const cacheFileSize = await CacheManager.getSize(cacheKey);
    const skipCache = opts?.useCache === false;

    let size = 0;
		let contentType = '';
		let supportRange = false;
		if(window.internet === false && cacheFileSize){
			size = cacheFileSize;
		}
		else{
			const response = await customFetch(url, { method: 'HEAD' });
      size = Number(response.headers.get('content-length'));
	    contentType = response.headers.get('content-type') || '';
	    supportRange = response.headers.get('accept-ranges') === 'bytes';
    }
    
    if (size !== 0 && size === cacheFileSize && !skipCache) {
      opts?.logger?.debug(`Using cached file ${cacheKey}`);
      const cachedFile = await CacheManager.open(cacheKey);
      (opts?.startSignal ?? Promise.resolve()).then(() => {
        opts?.progressCallback?.({
          loaded: cacheFileSize,
          total: cacheFileSize,
        });
      });
      return new GGUFRemoteBlob(url, 0, cacheFileSize, '', true, customFetch, {
        cachedStream: cachedFile!,
        progressCallback: () => {}, // unused
      });
    } else {
      if (cacheFileSize > 0) {
        opts?.logger?.debug(`Cache file is present, but size mismatch (cache = ${cacheFileSize} bytes, real = ${size} bytes)`);
      }
      opts?.logger?.debug(`NOT using cache for ${cacheKey}`);
      return new GGUFRemoteBlob(url, 0, size, contentType, true, customFetch, {
        progressCallback: opts?.progressCallback ?? (() => {}),
        startSignal: opts?.startSignal,
      });
    }
  }

  private url: RequestInfo | URL;
  private start: number;
  private end: number;
  private contentType: string;
  private full: boolean;
  private fetch: typeof fetch;
  private cachedStream?: ReadableStream;
  private progressCallback: ProgressCallback;
  private startSignal?: Promise<void>;

  constructor(url: RequestInfo | URL, start: number, end: number, contentType: string, full: boolean, customFetch: typeof fetch, additionals: {
    cachedStream?: ReadableStream,
    progressCallback: ProgressCallback,
    startSignal?: Promise<void>,
  }) {
    super([]);

    this.url = url;
    this.start = start;
    this.end = end;
    this.contentType = contentType;
    this.full = full;
    this.fetch = customFetch;
    this.cachedStream = additionals.cachedStream;
    this.progressCallback = additionals.progressCallback;
    this.startSignal = additionals.startSignal;
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
        controller.enqueue(chunk);
        loaded += chunk.byteLength;
        self.progressCallback({
          loaded,
          total: self.size,
        });
      },
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
          CacheManager.write(this.url.toString(), src1);
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
