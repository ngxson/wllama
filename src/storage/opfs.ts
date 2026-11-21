import { createWorker, isSafariMobile } from '../utils';
import { OPFS_UTILS_WORKER_CODE } from '../workers-code/generated';
import type { StorageBackend, StorageFileHint } from './index';

// the handle of each File read from the cache: the worker reads it with a sync access handle, see "Async file read" in README-dev.md
const fileHandles = new WeakMap<Blob, FileSystemFileHandle>();
export const getOPFSFileHandle = (blob: Blob) => fileHandles.get(blob);

export class OPFSBackend implements StorageBackend {
  isSupported(): boolean {
    return (
      typeof navigator !== 'undefined' &&
      'storage' in navigator &&
      !!navigator.storage?.getDirectory
    );
  }

  async read(key: string): Promise<Blob | null> {
    try {
      const cacheDir = await getCacheDir();
      const fileHandle = await cacheDir.getFileHandle(key);
      const file = await fileHandle.getFile();
      fileHandles.set(file, fileHandle);
      return file;
    } catch (e) {
      // NotFoundError or similar
      return null;
    }
  }

  async write(
    key: string,
    stream: ReadableStream,
    _hint?: StorageFileHint,
    offset: number = 0
  ): Promise<void> {
    const writable = await openWritable(key, offset);
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

  canWriteAt(): boolean {
    return true;
  }

  async getSize(key: string): Promise<number> {
    try {
      const cacheDir = await getCacheDir();
      const fileHandle = await cacheDir.getFileHandle(key);
      const file = await fileHandle.getFile();
      return file.size;
    } catch (e) {
      return -1;
    }
  }

  async list(): Promise<Array<{ key: string; size: number }>> {
    const cacheDir = await getCacheDir();
    const result: Array<{ key: string; size: number }> = [];
    // @ts-ignore
    for await (const [name, handle] of cacheDir.entries()) {
      if (handle.kind === 'file') {
        const file = await (handle as FileSystemFileHandle).getFile();
        result.push({ key: name, size: file.size });
      }
    }
    return result;
  }

  async delete(key: string): Promise<void> {
    try {
      const cacheDir = await getCacheDir();
      await cacheDir.removeEntry(key);
    } catch (e: any) {
      if (e?.name !== 'NotFoundError') throw e;
    }
  }
}

async function getCacheDir(): Promise<FileSystemDirectoryHandle> {
  const opfsRoot = await navigator.storage.getDirectory();
  return opfsRoot.getDirectoryHandle('cache', { create: true });
}

async function openWritable(
  fileName: string,
  offset: number
): Promise<{
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
  worker.onerror = (e) => pReject?.(e.message ?? e);
  const workerExec = (
    data:
      | { action: 'open'; filename: string; offset: number }
      | { action: 'write'; buf: Uint8Array }
      | { action: 'close' }
  ) =>
    new Promise<void>((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile()
          ? undefined
          : { transfer: 'buf' in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
  await workerExec({ action: 'open', filename: fileName, offset });
  return {
    write: (value) => workerExec({ action: 'write', buf: value }),
    close: async () => {
      await workerExec({ action: 'close' });
      worker.terminate();
    },
  };
}
