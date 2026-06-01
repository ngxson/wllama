export interface StorageFileHint {
  sha256: string;
}

export interface StorageBackend {
  isSupported(): boolean;

  /**
   * Read a file from storage by key.
   * @returns Blob, or null if the key does not exist
   */
  read(key: string, hint?: StorageFileHint): Promise<Blob | null>;

  /**
   * Write a ReadableStream to storage under the given key.
   * Overwrites any existing content for that key.
   */
  write(
    key: string,
    stream: ReadableStream,
    hint?: StorageFileHint
  ): Promise<void>;

  /**
   * Get the stored size of a file in bytes.
   * @returns number of bytes, or -1 if the key does not exist
   */
  getSize(key: string, hint?: StorageFileHint): Promise<number>;

  /**
   * List all keys currently in storage.
   * Includes metadata keys - callers are responsible for filtering.
   */
  list(): Promise<Array<{ key: string; size: number }>>;

  /**
   * Delete a single entry by key. No-op if the key does not exist.
   */
  delete(key: string): Promise<void>;
}
