export interface StorageFileHint {
  sha256: string;
}

export interface StorageWriter {
  write(pos: number, buf: Uint8Array): Promise<void>;
  close(): Promise<void>;
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
   * With `offset > 0`, keep the first `offset` bytes and write after them (only if `canWriteAt()` is true).
   */
  write(
    key: string,
    stream: ReadableStream,
    hint?: StorageFileHint,
    offset?: number
  ): Promise<void>;

  /**
   * True if `write()` accepts an offset, used to resume a download
   */
  canWriteAt?(hint?: StorageFileHint): boolean;

  /**
   * Random-access writer, keeps the first `offset` bytes. Only if `canWriteAt()` is true
   */
  openWriter?(
    key: string,
    offset: number,
    hint?: StorageFileHint
  ): Promise<StorageWriter>;

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
