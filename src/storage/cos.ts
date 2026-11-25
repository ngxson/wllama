import type { StorageBackend, StorageFileHint, StorageWriter } from './index';
import { OPFSBackend } from './opfs';

interface CrossOriginStorageGetFileHandleHash {
  value: string;
  algorithm: string;
}

interface CrossOriginStorageManager {
  getFileHandle(
    hash: CrossOriginStorageGetFileHandleHash,
    options?: { create?: boolean; origins?: string[] | string }
  ): Promise<FileSystemFileHandle>;
}

declare global {
  interface Navigator {
    readonly crossOriginStorage?: CrossOriginStorageManager;
  }
}

function makeHash(key: string): CrossOriginStorageGetFileHandleHash {
  return { algorithm: 'SHA-256', value: key };
}

// internal, non-standard implementation
class COSInternalBackend implements StorageBackend {
  // set when the Permissions Policy blocks COS in this document (for example, an iframe without allow="cross-origin-storage")
  private notAllowed = false;

  isSupported(): boolean {
    return (
      !this.notAllowed &&
      typeof navigator !== 'undefined' &&
      typeof navigator.crossOriginStorage?.getFileHandle === 'function'
    );
  }

  // IMPORTANT: key must be SHA-256 hash of the data
  async read(key: string): Promise<Blob | null> {
    try {
      const handle = await navigator.crossOriginStorage!.getFileHandle(
        makeHash(key)
      );
      return handle.getFile();
    } catch (e) {
      this.checkNotAllowed(e);
      return null;
    }
  }

  // IMPORTANT: key must be SHA-256 hash of the data
  async write(key: string, stream: ReadableStream): Promise<void> {
    const handle = await navigator
      .crossOriginStorage!.getFileHandle(makeHash(key), {
        create: true,
        origins: '*',
      })
      .catch((e) => {
        this.checkNotAllowed(e);
        throw e;
      });
    const writable = await (handle as any).createWritable();
    const reader = stream.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        await writable.write(value);
      }
    } finally {
      await writable.close();
    }
  }

  // IMPORTANT: key must be SHA-256 hash of the data
  async getSize(key: string): Promise<number> {
    try {
      const handle = await navigator.crossOriginStorage!.getFileHandle(
        makeHash(key)
      );
      const file = await handle.getFile();
      return file.size;
    } catch (e) {
      this.checkNotAllowed(e);
      return -1;
    }
  }

  private checkNotAllowed(e: any): void {
    if (e?.name === 'NotAllowedError') this.notAllowed = true;
  }

  async list(): Promise<Array<{ key: string; size: number }>> {
    throw new Error('not implemented');
  }

  async delete(_key: string): Promise<void> {
    throw new Error('not implemented');
  }
}

/**
 * Storage backend that uses the Cross-Origin Storage API
 * Metadata is stored in OPFS, while the actual data is stored in COS
 * If hint.sha256 is provided, it will be used as the key for COS, otherwise fallback to OPFS
 */
export class COSBackend implements StorageBackend {
  private cos = new COSInternalBackend();
  private priv = new OPFSBackend();

  isSupported(): boolean {
    return this.priv.isSupported();
  }

  async read(key: string, hint?: StorageFileHint): Promise<Blob | null> {
    if (hint?.sha256 && this.cos.isSupported()) {
      const blob = await this.cos.read(hint.sha256);
      if (blob) return blob;
    }
    return this.priv.read(key);
  }

  async write(
    key: string,
    stream: ReadableStream,
    hint?: StorageFileHint,
    offset: number = 0
  ): Promise<void> {
    if (hint?.sha256 && this.cos.isSupported()) {
      if (offset > 0) throw new Error('COS backend cannot write at an offset');
      try {
        await this.cos.write(hint.sha256, stream);
        return;
      } catch (e) {
        // the stream is not read yet, so OPFS can still use it
        if (this.cos.isSupported()) throw e;
      }
    }
    await this.priv.write(key, stream, undefined, offset);
  }

  canWriteAt(hint?: StorageFileHint): boolean {
    return !(hint?.sha256 && this.cos.isSupported());
  }

  openWriter(
    key: string,
    offset: number,
    hint?: StorageFileHint
  ): Promise<StorageWriter> {
    if (!this.canWriteAt(hint)) {
      throw new Error('COS backend cannot write at an offset');
    }
    return this.priv.openWriter(key, offset);
  }

  async getSize(key: string, hint?: StorageFileHint): Promise<number> {
    if (hint?.sha256 && this.cos.isSupported()) {
      const size = await this.cos.getSize(hint.sha256);
      if (size !== -1) return size;
    }
    return this.priv.getSize(key);
  }

  async list(): Promise<Array<{ key: string; size: number }>> {
    return this.priv.list();
  }

  async delete(key: string): Promise<void> {
    return this.priv.delete(key);
  }
}

// used for testing only
export function mockCOS(): void {
  const store = new Map<string, Blob>();

  (navigator as any).crossOriginStorage = {
    async getFileHandle(
      { value }: CrossOriginStorageGetFileHandleHash,
      options?: { create?: boolean }
    ): Promise<FileSystemFileHandle> {
      if (!options?.create && !store.has(value)) {
        throw new DOMException('File not found', 'NotFoundError');
      }
      return {
        getFile() {
          const blob = store.get(value);
          if (!blob) throw new DOMException('File not found', 'NotFoundError');
          return Promise.resolve(new File([blob], value));
        },
        createWritable() {
          const chunks: BlobPart[] = [];
          return Promise.resolve({
            write(chunk: BlobPart) {
              chunks.push(chunk);
              return Promise.resolve();
            },
            close() {
              store.set(value, new Blob(chunks));
              return Promise.resolve();
            },
          });
        },
      } as unknown as FileSystemFileHandle;
    },
  } satisfies CrossOriginStorageManager;
}
