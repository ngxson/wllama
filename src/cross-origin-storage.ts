/// <reference path="./cross-origin-storage-types.d.ts" />

/**
 * Cross-Origin Storage (COS) progressive enhancement.
 *
 * COS allows files identified by SHA-256 hash to be shared across origins, so a model
 * downloaded by one site is instantly available to any other site that opts in.
 *
 * Spec: https://github.com/WICG/cross-origin-storage
 * Extension: https://chromewebstore.google.com/detail/cross-origin-storage/denpnpcgjgikjpoglpjefakmdcbmlgih
 */

const COS_HASH_CACHE_NAME = 'wllama-cos-hash-cache';
const HASH_ALGORITHM = 'SHA-256';

function makeHashDescriptor(
  value: string
): CrossOriginStorageRequestFileHandleHash {
  return { algorithm: HASH_ALGORITHM, value };
}

/**
 * Returns true when the experimental Cross-Origin Storage API is available in
 * the current browser context (requires the COS extension or native support).
 */
export function isCrossOriginStorageAvailable(): boolean {
  return (
    typeof navigator !== 'undefined' && 'crossOriginStorage' in navigator
  );
}

/**
 * Resolve the SHA-256 hash for a URL.
 *
 * Strategy (cheapest first):
 * 1. Return the previously-cached hash from the wllama-cos-hash-cache Cache API bucket.
 * 2. Fetch the Hugging Face Git LFS pointer file (only for /resolve/ URLs) and extract
 *    the `oid sha256:` field.  This avoids reading the model file itself.
 * 3. Return null – no hash available without full-file download.
 */
export async function resolveUrlHash(url: string): Promise<string | null> {
  const cached = await getHashFromCache(url);
  if (cached) return cached;

  const lfsHash = await getLfsHash(url);
  if (lfsHash) {
    await setHashInCache(url, lfsHash);
    return lfsHash;
  }

  return null;
}

/**
 * Persist a URL → SHA-256 hash mapping so that future calls to `resolveUrlHash`
 * return immediately without a network round-trip.
 */
export async function setUrlHash(url: string, hash: string): Promise<void> {
  await setHashInCache(url, hash);
}

/**
 * Attempt to retrieve a file from Cross-Origin Storage by its SHA-256 hash.
 *
 * Returns `null` when:
 * - the COS API is unavailable
 * - the file is not present (or is privacy-gated by the browser)
 * - any unexpected error occurs
 */
export async function tryReadFromCOS(hash: string): Promise<Blob | null> {
  if (!isCrossOriginStorageAvailable()) return null;
  try {
    const [handle] = await navigator.crossOriginStorage.requestFileHandles([
      makeHashDescriptor(hash),
    ]);
    return handle.getFile();
  } catch {
    return null;
  }
}

/**
 * Write a `Blob` to Cross-Origin Storage under the given SHA-256 hash.
 *
 * The browser verifies that the hash of the written data matches `hash`.
 * Errors are swallowed – a COS write failure must never surface to the caller
 * because the file is already safely stored in OPFS.
 */
export async function writeToCosByHash(
  blob: Blob,
  hash: string
): Promise<void> {
  if (!isCrossOriginStorageAvailable()) return;
  try {
    const nav = navigator as any;
    const [handle]: FileSystemFileHandle[] =
      await nav.crossOriginStorage.requestFileHandles(
        [makeHashDescriptor(hash)],
        { create: true }
      );
    const writable = await (handle as any).createWritable();
    await writable.write(blob);
    await writable.close();
  } catch {
    // Non-fatal: COS write failure must not affect the calling code.
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function getHashFromCache(url: string): Promise<string | null> {
  try {
    const cache = await caches.open(COS_HASH_CACHE_NAME);
    const resp = await cache.match(url);
    return resp ? resp.text() : null;
  } catch {
    return null;
  }
}

async function setHashInCache(url: string, hash: string): Promise<void> {
  try {
    const cache = await caches.open(COS_HASH_CACHE_NAME);
    await cache.put(url, new Response(hash));
  } catch {
    // Cache API may be unavailable in some contexts (e.g. non-secure origins).
  }
}

/**
 * Fetch the SHA-256 hash from a Hugging Face Git LFS pointer file.
 *
 * HF stores large files in Git LFS. The `/raw/` endpoint returns the raw LFS
 * pointer (a small text file) instead of the actual blob, so we can learn the
 * content hash without downloading the multi-gigabyte model.
 *
 * Only works for URLs that contain `/resolve/` (the standard HF download path).
 * Returns null for non-HF or non-LFS URLs.
 */
async function getLfsHash(url: string): Promise<string | null> {
  if (!url.includes('/resolve/')) return null;
  // Swap /resolve/ for /raw/ to reach the LFS pointer; keep auth query params.
  const rawUrl = url.replace('/resolve/', '/raw/');
  try {
    const text = await fetch(rawUrl).then((r) => r.text());
    // LFS pointer format: "oid sha256:<hex>\n"
    const match = text.match(/^oid sha256:([0-9a-f]+)$/m);
    return match ? match[1] : null;
  } catch {
    return null;
  }
}
