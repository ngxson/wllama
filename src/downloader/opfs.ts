export async function opfsWrite(key: string, stream: ReadableStream): Promise<void> {
  const cacheDir = await getCacheDir();
  const fileName = await toFileName(key);
  const fileHandler = await cacheDir.getFileHandle(fileName, { create: true });
  const writable = await fileHandler.createWritable();
  await writable.truncate(0); // clear file content
  const reader = stream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    await writable.write(value);
  }
  await writable.close();
}

export async function opfsOpen(key: string): Promise<ReadableStream | null> {
  const cacheDir = await getCacheDir();
  const fileName = await toFileName(key);
  try {
    const fileHandler = await cacheDir.getFileHandle(fileName);
    const file = await fileHandler.getFile();
    return file.stream();
  } catch (e) {
    // TODO: check if exception is NotFoundError
    return null;
  }
}

export async function opfsFileSize(key: string): Promise<number> {
  const cacheDir = await getCacheDir();
  const fileName = await toFileName(key);
  try {
    const fileHandler = await cacheDir.getFileHandle(fileName);
    const file = await fileHandler.getFile();
    return file.size;
  } catch (e) {
    // TODO: check if exception is NotFoundError
    return -1;
  }
}

export async function opfsClear(): Promise<void> {
  const cacheDir = await getCacheDir();
  // @ts-ignore
  for await (let [name] of cacheDir.entries()) {
    cacheDir.removeEntry(name);
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
