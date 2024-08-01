import { createContext, useContext, useMemo, useState } from 'react';
import { getDefaultScreen, useDidMount, WllamaStorage } from './utils';
import { Wllama } from '@wllama/wllama';
import {
  DEFAULT_INFERENCE_PARAMS,
  LIST_MODELS,
  WLLAMA_CONFIG_PATHS,
} from '../config';
import {
  InferenceParams,
  ManageModel,
  Model,
  ModelState,
  Screen,
} from './types';
import { verifyCustomModel } from './custom-models';

interface WllamaContextValue {
  // functions for managing models
  models: ManageModel[];
  downloadModel(model: ManageModel): Promise<void>;
  removeModel(model: ManageModel): Promise<void>;
  removeAllModels(): Promise<void>;
  isDownloading: boolean;
  isLoadingModel: boolean;
  currModel?: ManageModel;
  loadModel(model: ManageModel): Promise<void>;
  unloadModel(): Promise<void>;
  currParams: InferenceParams;
  setParams(params: InferenceParams): void;

  // function for managing custom user model
  addCustomModel(url: string): Promise<void>;
  removeCustomModel(model: ManageModel): Promise<void>;

  // functions for chat completion
  getWllamaInstance(): Wllama;
  createCompletion(
    input: string,
    callback: (piece: string) => void
  ): Promise<void>;
  stopCompletion(): void;
  isGenerating: boolean;
  currentConvId: number;

  // nagivation
  navigateTo(screen: Screen, conversationId?: number): void;
  currScreen: Screen;
}

const WllamaContext = createContext<WllamaContextValue>({} as any);

let wllamaInstance = new Wllama(WLLAMA_CONFIG_PATHS);
let stopSignal = false;
const resetWllamaInstance = () => {
  wllamaInstance = new Wllama(WLLAMA_CONFIG_PATHS);
};

const getManageModels = async (): Promise<ManageModel[]> => {
  // TODO: remove "abandoned" files
  const cachedFiles = (await wllamaInstance.cacheManager.list()).filter((m) => {
    // remove files with sizes not matching remote
    return m.size === m.metadata.originalSize;
  });
  const cachedURLs = new Set(cachedFiles.map((e) => e.metadata.originalURL));
  const models = [...LIST_MODELS, ...WllamaStorage.load('custom_models', [])];
  return models.map((m) => ({
    ...m,
    name:
      m.url
        .split('/')
        .pop()
        ?.replace(/-\d{5}-of-\d{5}/, '')
        .replace('.gguf', '') ?? '(unknown)',
    state: cachedURLs.has(m.url) ? ModelState.READY : ModelState.NOT_DOWNLOADED,
    downloadPercent: 0,
  }));
};

export const WllamaProvider = ({ children }: any) => {
  const [isGenerating, setGenerating] = useState(false);
  const [currentConvId, setCurrentConvId] = useState(-1);
  const [currScreen, setScreen] = useState<Screen>(getDefaultScreen());
  const [models, setModels] = useState<ManageModel[]>([]);
  const [isBusy, setBusy] = useState(false);
  const [currParams, setCurrParams] = useState<InferenceParams>(
    WllamaStorage.load('params', DEFAULT_INFERENCE_PARAMS)
  );

  useDidMount(async () => {
    setModels(await getManageModels());
  });

  // computed variables
  const isDownloading = useMemo(
    () => models.some((m) => m.state === ModelState.DOWNLOADING),
    [models]
  );
  const isLoadingModel = useMemo(
    () => isBusy || models.some((m) => m.state === ModelState.LOADING),
    [models, isBusy]
  );
  const currModel = useMemo(
    () => models.find((m) => m.state === ModelState.LOADED),
    [models]
  );

  // utils
  const editModel = (newModel: ManageModel) =>
    setModels((models) =>
      models.map((m) => (m.url === newModel.url ? newModel : m))
    );
  const reloadModels = async () => {
    setModels(await getManageModels());
  };

  const downloadModel = async (model: ManageModel) => {
    if (isDownloading || currModel || isLoadingModel) return;
    editModel({ ...model, state: ModelState.DOWNLOADING, downloadPercent: 0 });
    try {
      await wllamaInstance.downloadModel(model.url, {
        progressCallback(opts) {
          editModel({
            ...model,
            state: ModelState.DOWNLOADING,
            downloadPercent: opts.loaded / opts.total,
          });
        },
      });
      editModel({ ...model, state: ModelState.READY, downloadPercent: 0 });
    } catch (e) {
      alert((e as any)?.message || 'unknown error while downloading model');
    }
  };

  const removeModel = async (model: ManageModel) => {
    const cacheKey = await wllamaInstance.cacheManager.getNameFromURL(
      model.url
    );
    await wllamaInstance.cacheManager.delete(cacheKey);
    await reloadModels();
  };

  const removeAllModels = async () => {
    await wllamaInstance.cacheManager.deleteMany(() => true);
    await reloadModels();
  };

  const loadModel = async (model: ManageModel) => {
    if (isDownloading || currModel || isLoadingModel) return;
    // if this is custom model, we make sure that it's up-to-date
    if (model.userAdded) {
      await downloadModel(model);
    }
    // make sure the model is cached
    if ((await wllamaInstance.cacheManager.getSize(model.url)) <= 0) {
      throw new Error('Model is not in cache');
    }
    editModel({ ...model, state: ModelState.LOADING, downloadPercent: 0 });
    try {
      await wllamaInstance.loadModelFromUrl(model.url, {
        n_threads: currParams.nThreads > 0 ? currParams.nThreads : undefined,
        n_ctx: currParams.nContext,
        n_batch: currParams.nBatch,
      });
      editModel({ ...model, state: ModelState.LOADED, downloadPercent: 0 });
    } catch (e) {
      resetWllamaInstance();
      alert(`Failed to load model: ${(e as any).message ?? 'Unknown error'}`);
      editModel({ ...model, state: ModelState.READY, downloadPercent: 0 });
    }
  };

  const unloadModel = async () => {
    if (!currModel) return;
    await wllamaInstance.exit();
    resetWllamaInstance();
    editModel({ ...currModel, state: ModelState.READY, downloadPercent: 0 });
  };

  const createCompletion = async (
    input: string,
    callback: (currentText: string) => void
  ) => {
    if (isDownloading || !currModel || isLoadingModel) return;
    setGenerating(true);
    stopSignal = false;
    const result = await wllamaInstance.createCompletion(input, {
      nPredict: currParams.nPredict,
      sampling: {
        temp: currParams.temperature,
      },
      // @ts-ignore unused variable
      onNewToken(token, piece, currentText, optionals) {
        callback(currentText);
        if (stopSignal) optionals.abortSignal();
      },
    });
    callback(result);
    stopSignal = false;
    setGenerating(false);
  };

  const stopCompletion = () => {
    stopSignal = true;
  };

  const navigateTo = (screen: Screen, conversationId?: number) => {
    setScreen(screen);
    setCurrentConvId(conversationId ?? -1);
    if (screen === Screen.MODEL) {
      WllamaStorage.save('welcome', false);
    }
  };

  // proxy function for saving to localStorage
  const setParams = (val: InferenceParams) => {
    WllamaStorage.save('params', val);
    setCurrParams(val);
  };

  // function for managing custom user model
  const addCustomModel = async (url: string) => {
    setBusy(true);
    try {
      const custom = await verifyCustomModel(url);
      if (models.some((m) => m.url === custom.url)) {
        throw new Error('Model with the same URL already exist');
      }
      const currList: Model[] = WllamaStorage.load('custom_models', []);
      WllamaStorage.save('custom_models', [...currList, custom]);
      await reloadModels();
    } catch (e) {
      setBusy(false);
      throw e; // re-throw
    }
    setBusy(false);
  };

  const removeCustomModel = async (model: ManageModel) => {
    setBusy(true);
    await removeModel(model);
    const currList: Model[] = WllamaStorage.load('custom_models', []);
    WllamaStorage.save(
      'custom_models',
      currList.filter((m) => m.url !== model.url)
    );
    await reloadModels();
    setBusy(false);
  };

  return (
    <WllamaContext.Provider
      value={{
        models,
        isDownloading,
        isLoadingModel,
        downloadModel,
        removeModel,
        removeAllModels,
        currModel,
        loadModel,
        unloadModel,
        currParams,
        setParams,
        createCompletion,
        stopCompletion,
        isGenerating,
        currentConvId,
        navigateTo,
        currScreen,
        getWllamaInstance: () => wllamaInstance,
        addCustomModel,
        removeCustomModel,
      }}
    >
      {children}
    </WllamaContext.Provider>
  );
};

export const useWllama = () => useContext(WllamaContext);
