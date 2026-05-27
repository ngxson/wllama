/**
 * TypeScript ambient declarations for the Cross-Origin Storage API.
 * Source: https://github.com/WICG/cross-origin-storage/blob/main/cross-origin-storage.d.ts
 */

interface CrossOriginStorageRequestFileHandleHash {
  value: string;
  algorithm: string;
}

interface CrossOriginStorageRequestFileHandleOptions {
  create?: boolean;
  origins?: string[] | string;
}

interface CrossOriginStorageManager {
  requestFileHandles(
    hashes: CrossOriginStorageRequestFileHandleHash[],
    options?: CrossOriginStorageRequestFileHandleOptions
  ): Promise<FileSystemFileHandle[]>;
}

interface Navigator {
  readonly crossOriginStorage: CrossOriginStorageManager;
}

interface WorkerNavigator {
  readonly crossOriginStorage: CrossOriginStorageManager;
}
