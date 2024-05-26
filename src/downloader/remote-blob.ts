// Adapted from https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/utils/WebBlob.ts

import { opfsFileSize, opfsOpen, opfsWrite } from './opfs';


/**
 * WebBlob is a Blob implementation for web resources that supports range requests.
 */

interface GGUFRemoteBlobCreateOptions {
  /**
   * Custom fetch function to use instead of the default one, for example to use a proxy or edit headers.
   */
  fetch?: typeof fetch;
  useCache?: boolean,
  /**
   * Custom debug logger
   */
  logger?: {
    debug: typeof console['debug'],
  };
}

export class GGUFRemoteBlob extends Blob {
  static async create(url: URL, opts?: GGUFRemoteBlobCreateOptions): Promise<Blob> {
    const customFetch = opts?.fetch ?? fetch;
    const response = await customFetch(url, { method: 'HEAD' });

    const size = Number(response.headers.get('content-length'));
    const contentType = response.headers.get('content-type') || '';
    const supportRange = response.headers.get('accept-ranges') === 'bytes';

    const cacheKey = url.toString();
    const cacheFileSize = await opfsFileSize(cacheKey);
    const skipCache = opts?.useCache === false;

    if (size === cacheFileSize && !skipCache) {
      opts?.logger?.debug(`Using cached file ${cacheKey}`);
      const cachedFile = await opfsOpen(cacheKey);
      return new GGUFRemoteBlob(url, 0, cacheFileSize, '', true, customFetch, cachedFile!);
    } else {
      if (cacheFileSize > 0) {
        opts?.logger?.debug(`Cache file is present, but size mismatch (cache = ${cacheFileSize} bytes, real = ${size} bytes)`);
      }
      opts?.logger?.debug(`NOT using cache for ${cacheKey}`);
      return new GGUFRemoteBlob(url, 0, size, contentType, true, customFetch);
    }
  }

  private url: URL;
  private start: number;
  private end: number;
  private contentType: string;
  private full: boolean;
  private fetch: typeof fetch;
  private cachedStream?: ReadableStream;

  constructor(url: URL, start: number, end: number, contentType: string, full: boolean, customFetch: typeof fetch, cachedStream?: ReadableStream) {
    super([]);

    this.url = url;
    this.start = start;
    this.end = end;
    this.contentType = contentType;
    this.full = full;
    this.fetch = customFetch;
    this.cachedStream = cachedStream;
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

    const stream = new TransformStream();

    this.fetchRange()
      .then((response) => {
        const [src0, src1] = response.body!.tee();
        src0.pipeThrough(stream);
        opfsWrite(this.url.toString(), src1);
      })
      .catch((error) => stream.writable.abort(error.message));

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
