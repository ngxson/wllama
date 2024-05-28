import { isSafari } from '../utils';

/**
 * Write to OPFS file from ReadableStream
 */
export async function opfsWrite(key: string, stream: ReadableStream): Promise<void> {
  try {
    const cacheDir = await getCacheDir();
    const fileName = await toFileName(key);
    const writable = isSafari()
      ? await opfsWriteViaWorker(fileName)
      : await cacheDir.getFileHandle(fileName, { create: true }).then(h => h.createWritable());
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
export async function opfsOpen(key: string): Promise<ReadableStream | null> {
  try {
    const cacheDir = await getCacheDir();
    const fileName = await toFileName(key);
    const fileHandler = await cacheDir.getFileHandle(fileName);
    const file = await fileHandler.getFile();
    return file.stream();
  } catch (e) {
    // TODO: check if exception is NotFoundError
    return null;
  }
}

/**
 * Get file size of a file in OPFS
 * @returns number of bytes, or -1 if file does not exist
 */
export async function opfsFileSize(key: string): Promise<number> {
  try {
    const cacheDir = await getCacheDir();
    const fileName = await toFileName(key);
    const fileHandler = await cacheDir.getFileHandle(fileName);
    const file = await fileHandler.getFile();
    return file.size;
  } catch (e) {
    // TODO: check if exception is NotFoundError
    return -1;
  }
}

/**
 * Clear everything in OPFS
 */
export async function opfsClear(): Promise<void> {
  try {
    const cacheDir = await getCacheDir();
    // @ts-ignore
    for await (let [name] of cacheDir.entries()) {
      cacheDir.removeEntry(name);
    }
  } catch (e) {
    // ignored
  }
}

async function toFileName(str: string) {
  const hashBuffer = await crypto.subtle.digest('SHA-1', new TextEncoder().encode(str));
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  return `${hashHex}_${str.split('/').pop()}`;
}

async function getCacheDir() {
  const opfsRoot = await navigator.storage.getDirectory();
  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });
  return cacheDir;
}

/**
 * Because safari does not support createWritable(), we need to use createSyncAccessHandle() which requires to be run from a web worker.
 * See: https://bugs.webkit.org/show_bug.cgi?id=231706
 */
const WORKER_CODE = `
const msg = (data) => postMessage(data);
let accessHandle;

onmessage = async (e) => {
  try {
    if (!e.data) return;
    const {
      open,  // name of file to open
      value, // value to be written
      done,  // indicates when to close the file
    } = e.data;

    if (open) {
      const opfsRoot = await navigator.storage.getDirectory();
      const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });
      const fileHandler = await cacheDir.getFileHandle(open, { create: true });
      accessHandle = await fileHandler.createSyncAccessHandle();
      accessHandle.truncate(0); // clear file content
      return msg({ ok: true });

    } else if (value) {
      accessHandle.write(value);
      return msg({ ok: true });

    } else if (done) {
      accessHandle.flush();
      accessHandle.close();
      return msg({ ok: true });
    }

    throw new Error('OPFS Worker: Invalid state');
  } catch (err) {
    return msg({ err });
  }
};
`;

async function opfsWriteViaWorker(fileName: string): Promise<{
  truncate(offset: number): Promise<void>,
  write(value: Uint8Array): Promise<void>,
  close(): Promise<void>,
}> {
  const workerURL = window.URL.createObjectURL(new Blob([WORKER_CODE], { type: 'text/javascript' }));
  const worker = new Worker(workerURL);
  let pResolve: (v: any) => void;
  let pReject: (v: any) => void;
  worker.onmessage = (e: MessageEvent<any>) => {
    if (e.data.ok) pResolve(null);
    else if (e.data.err) pReject(e.data.err);
  };
  const workerExec = (data: {
    open?: string,
    value?: Uint8Array,
    done?: boolean,
  }) => new Promise<void>((resolve, reject) => {
    pResolve = resolve;
    pReject = reject;
    worker.postMessage(data, data.value ? [data.value.buffer] : []);
  });
  await workerExec({ open: fileName });
  return {
    truncate: async () => { /* noop */ },
    write: (value) => workerExec({ value }),
    close: async () => {
      await workerExec({ done: true });
      worker.terminate();
    },
  };
};
