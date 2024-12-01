let accessHandle;
let abortController = new AbortController();

async function openFile(filename) {
  const opfsRoot = await navigator.storage.getDirectory();
  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });
  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });
  accessHandle = await fileHandler.createSyncAccessHandle();
  accessHandle.truncate(0); // clear file content
}

async function writeFile(buf) {
  accessHandle.write(buf);
}

async function closeFile() {
  accessHandle.flush();
  accessHandle.close();
}

async function writeTextFile(filename, str) {
  await openFile(filename);
  await writeFile(new TextEncoder().encode(str));
  await closeFile();
}

const throttled = (func, delay) => {
  let lastRun = 0;
  return (...args) => {
    const now = Date.now();
    if (now - lastRun > delay) {
      lastRun = now;
      func.apply(null, args);
    }
  };
};

const assertNonNull = (val) => {
  if (val === null || val === undefined) {
    throw new Error('OPFS Worker: Assertion failed');
  }
};

// respond to main thread
const resOK = () => postMessage({ ok: true });
const resProgress = (loaded, total) =>
  postMessage({ progress: { loaded, total } });
const resErr = (err) => postMessage({ err });

onmessage = async (e) => {
  try {
    if (!e.data) return;

    /**
     * @param {Object} e.data
     *
     * Fine-control FS actions:
     * - { action: 'open', filename: 'string' }
     * - { action: 'write', buf: ArrayBuffer }
     * - { action: 'close' }
     *
     * Simple write API:
     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }
     *
     * Download API:
     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }
     * - { action: 'download-abort' }
     */
    const { action, filename, buf, url, options, metadataFileName } = e.data;

    if (action === 'open') {
      assertNonNull(filename);
      await openFile(filename);
      return resOK();
    } else if (action === 'write') {
      assertNonNull(buf);
      await writeFile(buf);
      return resOK();
    } else if (action === 'close') {
      await closeFile();
      return resOK();
    } else if (action === 'write-simple') {
      assertNonNull(filename);
      assertNonNull(buf);
      await openFile(filename);
      await writeFile(buf);
      await closeFile();
      return resOK();
    } else if (action === 'download') {
      assertNonNull(url);
      assertNonNull(filename);
      assertNonNull(metadataFileName);
      assertNonNull(options);
      assertNonNull(options.aborted);
      abortController = new AbortController();
      if (options.aborted) abortController.abort();
      const response = await fetch(url, {
        ...options,
        signal: abortController.signal,
      });
      const contentLength = response.headers.get('content-length');
      const etag = (response.headers.get('etag') || '').replace(
        /[^A-Za-z0-9]/g,
        ''
      );
      const total = parseInt(contentLength, 10);
      const reader = response.body.getReader();
      await openFile(filename);
      let loaded = 0;
      const throttledProgress = throttled(resProgress, 100);
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        loaded += value.byteLength;
        await writeFile(value);
        throttledProgress(loaded, total);
      }
      resProgress(total, total); // 100% done
      await closeFile();
      // make sure this is in-sync with CacheEntryMetadata
      await writeTextFile(
        metadataFileName,
        JSON.stringify({
          originalURL: url,
          originalSize: total,
          etag,
        })
      );
      return resOK();
    } else if (action === 'download-abort') {
      if (abortController) {
        abortController.abort();
      }
      return;
    }

    throw new Error('OPFS Worker: Invalid action', e.data);
  } catch (err) {
    return resErr(err);
  }
};
