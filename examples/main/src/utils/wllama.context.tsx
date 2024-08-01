import { createContext, useContext, useMemo, useState } from 'react';
import { delay, useDidMount, WllamaStorage } from './utils';
import { Wllama } from '@wllama/wllama';
import { DEFAULT_INFERENCE_PARAMS, LIST_MODELS, WLLAMA_CONFIG_PATHS } from './config';
import { InferenceParams, ManageModel, Model, ModelState, Screen } from './types';

interface WllamaContextValue {
  // functions for managing models
  models: ManageModel[];
  downloadModel(model: ManageModel): Promise<void>;
  removeModel(model: ManageModel): Promise<void>;
  removeAllModels(): Promise<void>;
  isDownloading: boolean;
  isLoadingModel: boolean;
  currModel?: Model;
  loadModel(model: Model): Promise<void>;
  unloadModel(): Promise<void>;
  currParams: InferenceParams;
  setParams(params: InferenceParams): void;

  // functions for chat completion
  createCompletion(input: string, callback: (piece: string) => void): Promise<void>;
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

const getManageModels = async (): Promise<ManageModel[]> => {
  // TODO: remove "abandoned" files
  const cachedFiles = (await wllamaInstance.cacheManager.list()).filter(m => {
    // remove files with sizes not matching remote
    return m.size === m.metadata.originalSize;
  });
  const cachedURLs = new Set(cachedFiles.map(e => e.metadata.originalURL));
  return LIST_MODELS.map(m => ({
    ...m,
    state: cachedURLs.has(m.url) ? ModelState.READY : ModelState.NOT_DOWNLOADED,
    downloadPercent: 0,
  }));
};

export const WllamaProvider = ({ children }: any) => {
  const [isGenerating, setGenerating] = useState(false);
  const [currentConvId, setCurrentConvId] = useState(-1);
  const [currScreen, setScreen] = useState<Screen>(Screen.MODEL);
  const [models, setModels] = useState<ManageModel[]>([]);
  const [currParams, setCurrParams] = useState<InferenceParams>(
    WllamaStorage.load('params', DEFAULT_INFERENCE_PARAMS)
  );

  useDidMount(async () => {
    setModels(await getManageModels());
  });

  // computed variables
  const isDownloading = useMemo(() => models.some(m => m.state === ModelState.DOWNLOADING), [models]);
  const isLoadingModel = useMemo(() => models.some(m => m.state === ModelState.LOADING), [models]);
  const currModel = useMemo(() => models.find(m => m.state === ModelState.LOADED), [models]);

  // utils
  const editModel = (newModel: ManageModel) => setModels(models => models.map(m => m.url === newModel.url ? newModel : m));
  const reloadModels = async () => {
    setModels(await getManageModels());
  };

  const downloadModel = async (model: ManageModel) => {
    if (isDownloading || currModel || isLoadingModel) return;
    editModel({...model, state: ModelState.DOWNLOADING, downloadPercent: 0});
    try {
      await wllamaInstance.downloadModel(model.url, {
        progressCallback(opts) {
          editModel({...model, state: ModelState.DOWNLOADING, downloadPercent: opts.loaded / opts.total });
        },
      });
      editModel({...model, state: ModelState.READY, downloadPercent: 0});
    } catch (e) {
      alert((e as any)?.message || 'unknown error while downloading model');
    }
  };

  const removeModel = async (model: ManageModel) => {
    const cacheKey = await wllamaInstance.cacheManager.getNameFromURL(model.url);
    await wllamaInstance.cacheManager.delete(cacheKey);
    await reloadModels();
  };

  const removeAllModels = async () => {
    await wllamaInstance.cacheManager.deleteMany(() => true);
    await reloadModels();
  };

  const loadModel = async (model: Model) => {
    if (isDownloading || currModel || isLoadingModel) return;
    // make sure the model is cached
    if (await wllamaInstance.cacheManager.getSize(model.url) <= 0) {
      throw new Error('Model is not in cache');
    }
    editModel({...model, state: ModelState.LOADING, downloadPercent: 0});
    await wllamaInstance.loadModelFromUrl(model.url, {
      n_threads: currParams.nThreads > 0 ? currParams.nThreads : undefined,
      n_ctx: currParams.nContext,
    });
    editModel({...model, state: ModelState.LOADED, downloadPercent: 0});
    // TODO: handle exceptions
  };

  const unloadModel = async () => {
    if (!currModel) return;
    await wllamaInstance.exit();
    wllamaInstance = new Wllama(WLLAMA_CONFIG_PATHS);
    editModel({...currModel, state: ModelState.READY, downloadPercent: 0});
  };

  const createCompletion = async (input: string, callback: (currentText: string) => void) => {
    if (isDownloading || !currModel || isLoadingModel) return;
    setGenerating(true);
    stopSignal = false;
    const result = await wllamaInstance.createCompletion(input, {
      nPredict: currParams.nPredict,
      sampling: {
        temp: currParams.temperature,
      },
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
  };

  // proxy function for saving to localStorage
  const setParams = (val: InferenceParams) => {
    WllamaStorage.save('params', val);
    setCurrParams(val);
  };

  return (
    <WllamaContext.Provider value={{
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
    }}>{children}</WllamaContext.Provider>
  );
};

export const useWllama = () => useContext(WllamaContext);
