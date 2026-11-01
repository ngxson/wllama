let accessHandle;
let abortController = new AbortController();

async function openFile(filename, keep = false) {
  const opfsRoot = await navigator.storage.getDirectory();
  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });
  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });
  accessHandle = await fileHandler.createSyncAccessHandle();
  if (!keep) accessHandle.truncate(0); // clear file content
}

// Tries in a row with no bytes gained before a download fails.
const DOWNLOAD_ATTEMPTS = 6;

class HttpError extends Error {}

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
    const {
      action,
      filename,
      buf,
      url,
      options,
      metadataFileName,
      metadataAdditional,
    } = e.data;

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
      await openFile(filename);
      let loaded = 0;
      let total = NaN;
      let etag = '';
      const throttledProgress = throttled(resProgress, 100);
      // A dropped connection, or the browser closing the file mid-download
      // (Safari can), doesn't start over: it retries, asking only for the
      // bytes still missing, through a new handle if the old one was closed.
      for (let attempt = 1; ;) {
        const start = loaded;
        try {
          const headers = new Headers(options.headers || {});
          if (loaded > 0) headers.set('Range', 'bytes=' + loaded + '-');
          const response = await fetch(url, {
            ...options,
            headers,
            signal: abortController.signal,
          });
          if (!response.ok) {
            throw new HttpError('HTTP ' + response.status + ' for ' + url);
          }
          if (loaded > 0 && response.status !== 206) {
            // The server sent the whole file instead of the rest.
            accessHandle.truncate(0);
            loaded = 0;
          }
          if (loaded === 0) {
            total = parseInt(response.headers.get('content-length'), 10);
            etag = (response.headers.get('etag') || '').replace(
              /[^A-Za-z0-9]/g,
              ''
            );
          }
          const reader = response.body.getReader();
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              accessHandle.write(value, { at: loaded });
              loaded += value.byteLength;
              throttledProgress(loaded, total);
            }
          } catch (err) {
            // (A write failed: stop the response rather than let it run on.)
            reader.cancel().catch(() => {});
            throw err;
          }
          break;
        } catch (err) {
          if (abortController.signal.aborted || err instanceof HttpError) {
            throw err;
          }
          try {
            loaded = accessHandle.getSize();
          } catch {
            // The browser closed the file: a new handle, keeping what's there.
            try {
              accessHandle.close();
            } catch {}
            await openFile(filename, true);
            loaded = accessHandle.getSize();
          }
          attempt = loaded > start ? 1 : attempt + 1;
          if (attempt > DOWNLOAD_ATTEMPTS) throw err;
          await new Promise((resolve) =>
            setTimeout(resolve, 1000 * 2 ** (attempt - 1))
          );
        }
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
          ...metadataAdditional,
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
