import { Wllama } from '@wllama/wllama';
import { openDB, DBSchema, IDBPDatabase } from 'idb';

// See: https://vitejs.dev/guide/assets#explicit-url-imports
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';
import wllamaMultiWorker from '@wllama/wllama/src/multi-thread/wllama.worker.mjs?url';
import { createContext, useContext, useEffect, useMemo, useState } from 'react';

export const EMBD_MODEL_URL = 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q2_K.gguf';
export const EMBD_MODEL_NAME = 'nomic-embed-text-v1.5';

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.wasm': wllamaMulti,
  'multi-thread/wllama.worker.mjs': wllamaMultiWorker,
};

export const getWllamaConfigPaths = () => CONFIG_PATHS;

////////////////////////////////////////////////////////

export interface DataEntry {
  content: string;
  metadata: any;
  embeddings: number[];
};

interface Database extends DBSchema {
  chunks: {
    key: number;
    value: DataEntry;
  };
};

////////////////////////////////////////////////////////

const AppContext = createContext<{
  wllama: Wllama;
  isModelLoaded: boolean;
  dataset: DataEntry[];
  dbSet(newDataset: DataEntry[]): Promise<void>;
}>({} as any);

export const AppContextProvider = ({ children }: any) => {
  const wllama = useMemo(() => new Wllama(getWllamaConfigPaths()), []);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [dataset, setDataset] = useState<DataEntry[]>([]);
  const [db, setDB] = useState<IDBPDatabase<Database>>();

  useEffect(() => {
    (async () => {
      const db = await openDB<Database>('vectordb', 1, {
        async upgrade(db) {
          db.createObjectStore('chunks');
        },
      });
      setDB(db);
      const dataset = await db.getAll('chunks');
      setDataset(dataset);
      await wllama.loadModelFromUrl(EMBD_MODEL_URL, {
        n_batch: 2048,
        embeddings: true,
      });
      setIsModelLoaded(true);
    })();
  }, []);

  const dbSet = async (newDataset: DataEntry[]) => {
    await db?.clear('chunks');
    let i = 0;
    for (const d of newDataset) {
      await db?.put('chunks', d, i++);
    }
    setDataset(newDataset);
  };

  return <AppContext.Provider
    value={{
      wllama,
      isModelLoaded,
      dataset,
      dbSet,
    }}
  >
    {children}
  </AppContext.Provider>;
};

export const useAppContext = () => useContext(AppContext);

////////////////////////////////////////////////////////

/**
 * Download a string as file
 */
export function downloadStringAsFile(filename: string, text: string) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);
  element.style.display = 'none';
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
}
