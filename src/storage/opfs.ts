import { createWorker, isSafariMobile } from '../utils';
import { OPFS_UTILS_WORKER_CODE } from '../workers-code/generated';
import type { StorageBackend } from './index';

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
      return await fileHandle.getFile();
    } catch (e) {
      // NotFoundError or similar
      return null;
    }
  }

  async write(key: string, stream: ReadableStream): Promise<void> {
    const writable = await openWritable(key);
    await writable.truncate(0);
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

async function openWritable(fileName: string): Promise<{
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
  worker.onerror = (e) => pReject?.(e.message ?? e);
  const workerExec = (
    data:
      | { action: 'open'; filename: string }
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
  await workerExec({ action: 'open', filename: fileName });
  return {
    truncate: async () => {
      /* worker's openFile already calls truncate(0) on open */
    },
    write: (value) => workerExec({ action: 'write', buf: value }),
    close: async () => {
      await workerExec({ action: 'close' });
      worker.terminate();
    },
  };
}
