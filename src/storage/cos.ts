import type { StorageBackend, StorageFileHint } from './index';
import { OPFSBackend } from './opfs';

interface CrossOriginStorageRequestFileHandleHash {
  value: string;
  algorithm: string;
}

interface CrossOriginStorageManager {
  requestFileHandles(
    hashes: CrossOriginStorageRequestFileHandleHash[],
    options?: { create?: boolean; origins?: string[] | string }
  ): Promise<FileSystemFileHandle[]>;
}

declare global {
  interface Navigator {
    readonly crossOriginStorage: CrossOriginStorageManager;
  }
}

function makeHash(key: string): CrossOriginStorageRequestFileHandleHash {
  return { algorithm: 'SHA-256', value: key };
}

// internal, non-standard implementation
class COSInternalBackend implements StorageBackend {
  isSupported(): boolean {
    return (
      typeof navigator !== 'undefined' && 'crossOriginStorage' in navigator
    );
  }

  // IMPORTANT: key must be SHA-256 hash of the data
  async read(key: string): Promise<Blob | null> {
    try {
      const [handle] = await navigator.crossOriginStorage.requestFileHandles([
        makeHash(key),
      ]);
      return handle.getFile();
    } catch {
      return null;
    }
  }

  // IMPORTANT: key must be SHA-256 hash of the data
  async write(key: string, stream: ReadableStream): Promise<void> {
    const [handle] = await navigator.crossOriginStorage.requestFileHandles(
      [makeHash(key)],
      { create: true }
    );
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
      const [handle] = await navigator.crossOriginStorage.requestFileHandles([
        makeHash(key),
      ]);
      const file = await handle.getFile();
      return file.size;
    } catch {
      return -1;
    }
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
    hint?: StorageFileHint
  ): Promise<void> {
    if (hint?.sha256 && this.cos.isSupported()) {
      const [s1, s2] = stream.tee();
      await Promise.all([
        this.cos.write(hint.sha256, s1),
        this.priv.write(key, s2),
      ]);
    } else {
      await this.priv.write(key, stream);
    }
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
