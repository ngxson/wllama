import { createContext, useContext, useMemo, useState } from 'react';
import {
  DebugLogger,
  getDefaultScreen,
  useDidMount,
  WllamaStorage,
} from './utils';
import { Model, ModelManager, Wllama } from '@wllama/wllama';
import { DEFAULT_INFERENCE_PARAMS, WLLAMA_CONFIG_PATHS } from '../config';
import { InferenceParams, RuntimeInfo, ModelState, Screen } from './types';
import { verifyCustomModel } from './custom-models';
import {
  DisplayedModel,
  getDisplayedModels,
  getUserAddedModels,
  updateUserAddedModels,
} from './displayed-model';

interface WllamaContextValue {
  // functions for managing models
  models: DisplayedModel[];
  downloadModel(model: DisplayedModel): Promise<void>;
  removeCachedModel(model: DisplayedModel): Promise<void>;
  removeAllCachedModels(): Promise<void>;
  isDownloading: boolean;
  isLoadingModel: boolean;
  currParams: InferenceParams;
  setParams(params: InferenceParams): void;

  // function to load/unload model
  loadedModel?: DisplayedModel;
  currRuntimeInfo?: RuntimeInfo;
  loadModel(model: DisplayedModel): Promise<void>;
  unloadModel(): Promise<void>;

  // function for managing custom user model
  addCustomModel(url: string): Promise<void>;
  removeCustomModel(model: DisplayedModel): Promise<void>;

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

const modelManager = new ModelManager();
let wllamaInstance = new Wllama(WLLAMA_CONFIG_PATHS, { logger: DebugLogger });
let stopSignal = false;
const resetWllamaInstance = () => {
  wllamaInstance = new Wllama(WLLAMA_CONFIG_PATHS, { logger: DebugLogger });
};

export const WllamaProvider = ({ children }: any) => {
  const [isGenerating, setGenerating] = useState(false);
  const [currentConvId, setCurrentConvId] = useState(-1);
  const [currScreen, setScreen] = useState<Screen>(getDefaultScreen());
  const [cachedModels, setCachedModels] = useState<Model[]>([]);
  const [isBusy, setBusy] = useState(false);
  const [currRuntimeInfo, setCurrRuntimeInfo] = useState<RuntimeInfo>();
  const [currParams, setCurrParams] = useState<InferenceParams>(
    WllamaStorage.load('params', DEFAULT_INFERENCE_PARAMS)
  );
  const [downloadingProgress, setDownloadingProgress] = useState<
    Record<DisplayedModel['url'], number>
  >({});
  const [loadedModel, setLoadedModel] = useState<DisplayedModel>();

  const refreshCachedModels = async () => {
    setCachedModels(await modelManager.getModels());
  };
  useDidMount(refreshCachedModels);

  // computed variables
  const models = useMemo(() => {
    const list = getDisplayedModels(cachedModels);
    for (const model of list) {
      model.downloadPercent = downloadingProgress[model.url] ?? -1;
      if (model.downloadPercent >= 0) {
        model.state = ModelState.DOWNLOADING;
      }
      if (loadedModel?.url === model.url) {
        model.state = loadedModel.state;
      }
    }
    return list;
  }, [cachedModels, downloadingProgress, loadedModel]);
  const isDownloading = useMemo(
    () => models.some((m) => m.state === ModelState.DOWNLOADING),
    [models]
  );
  const isLoadingModel = useMemo(
    () => isBusy || loadedModel?.state === ModelState.LOADING,
    [loadedModel, isBusy]
  );

  // utils
  const updateModelDownloadState = (
    url: string,
    downloadPercent: number = -1
  ) => {
    if (downloadPercent < 0) {
      setDownloadingProgress((p) => {
        const newProgress = { ...p };
        delete newProgress[url];
        return newProgress;
      });
    } else {
      setDownloadingProgress((p) => ({ ...p, [url]: downloadPercent }));
    }
  };

  const downloadModel = async (model: DisplayedModel) => {
    if (isDownloading || loadedModel || isLoadingModel) return;
    updateModelDownloadState(model.url, 0);
    try {
      await modelManager.downloadModel(model.url, {
        progressCallback(opts) {
          updateModelDownloadState(model.url, opts.loaded / opts.total);
        },
      });
      updateModelDownloadState(model.url, -1);
      await refreshCachedModels();
    } catch (e) {
      alert((e as any)?.message || 'unknown error while downloading model');
    }
  };

  const removeCachedModel = async (model: DisplayedModel) => {
    if (isDownloading || loadedModel || isLoadingModel) return;
    if (model.cachedModel) {
      await model.cachedModel.remove();
      await refreshCachedModels();
    }
  };

  const removeAllCachedModels = async () => {
    if (isDownloading || loadedModel || isLoadingModel) return;
    await modelManager.clear();
    await refreshCachedModels();
  };

  const loadModel = async (model: DisplayedModel) => {
    if (isDownloading || loadedModel || isLoadingModel) return;
    // make sure the model is cached
    if (!model.cachedModel) {
      throw new Error('Model is not in cache');
    }
    setLoadedModel(model.clone({ state: ModelState.LOADING }));
    try {
      await wllamaInstance.loadModel(model.cachedModel, {
        n_threads: currParams.nThreads > 0 ? currParams.nThreads : undefined,
        n_ctx: currParams.nContext,
        n_batch: currParams.nBatch,
      });
      setLoadedModel(model.clone({ state: ModelState.LOADED }));
      setCurrRuntimeInfo({
        isMultithread: wllamaInstance.isMultithread(),
        hasChatTemplate: !!wllamaInstance.getChatTemplate(),
      });
    } catch (e) {
      resetWllamaInstance();
      alert(`Failed to load model: ${(e as any).message ?? 'Unknown error'}`);
      setLoadedModel(undefined);
    }
  };

  const unloadModel = async () => {
    if (!loadedModel) return;
    await wllamaInstance.exit();
    resetWllamaInstance();
    setLoadedModel(undefined);
    setCurrRuntimeInfo(undefined);
  };

  const createCompletion = async (
    input: string,
    callback: (currentText: string) => void
  ) => {
    if (isDownloading || !loadedModel || isLoadingModel) return;
    setGenerating(true);
    stopSignal = false;
    const result = await wllamaInstance.createCompletion(input, {
      nPredict: currParams.nPredict,
      useCache: true,
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
      const userAddedModels = getUserAddedModels(cachedModels);
      updateUserAddedModels([
        ...userAddedModels,
        new DisplayedModel(custom.url, custom.size, true, undefined),
      ]);
      await refreshCachedModels();
    } catch (e) {
      setBusy(false);
      throw e; // re-throw
    }
    setBusy(false);
  };

  const removeCustomModel = async (model: DisplayedModel) => {
    setBusy(true);
    if (model.isUserAdded) {
      const userAddedModels = getUserAddedModels(cachedModels);
      const newList = userAddedModels.filter((m) => m.url !== model.url);
      updateUserAddedModels(newList);
      await refreshCachedModels();
    } else {
      throw new Error('Cannot remove non-user-added model');
    }
    setBusy(false);
  };

  return (
    <WllamaContext.Provider
      value={{
        models,
        isDownloading,
        isLoadingModel,
        downloadModel,
        removeCachedModel,
        removeAllCachedModels,
        loadedModel,
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
        currRuntimeInfo,
      }}
    >
      {children}
    </WllamaContext.Provider>
  );
};

export const useWllama = () => useContext(WllamaContext);
