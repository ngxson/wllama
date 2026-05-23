import { ProxyToWorker, type WllamaWorkerResources } from './worker';
import {
  absoluteUrl,
  canUseAsyncFileRead,
  cbToAsyncIter,
  checkEnvironmentCompatible,
  isFirefox,
  isString,
  isSupportJSPI,
  isSupportMultiThread,
  isSupportWebGPU,
  MMPROJ_FILE_NAME,
  needCompat,
  prepareBlobs,
} from './utils';
import CacheManager, { type DownloadOptions } from './cache-manager';
import { ModelManager, Model, type ModelSource } from './model-manager';
import type {
  GlueMsgCompletionRes,
  GlueMsgEmbeddingRes,
  GlueMsgGetResultRes,
  GlueMsgLoadRes,
} from './glue/messages';
import { LIBLLAMA_VERSION } from './workers-code/generated';
import type {
  LoadedContextInfo,
  LoadModelParams,
  StreamParams,
} from './types/types';
import type {
  ChatCompletionChunk,
  ChatCompletionParams,
  ChatCompletionResponse,
  ChatCompletionUserMessage,
  CreateEmbeddingResponse,
  EmbeddingCreateParams,
  RawCompletionChunk,
  RawCompletionParams,
  RawCompletionResponse,
} from './types/oai-compat';
import { LogLevel } from './types/types';
import { getHFModelSource, type HuggingFaceParams } from './huggingface';
import { WasmCompatFromCDN } from './wasm-from-cdn';

export interface WllamaLogger {
  debug: typeof console.debug;
  log: typeof console.log;
  warn: typeof console.warn;
  error: typeof console.error;
}

// TODO: bring back useCache
export interface WllamaConfig {
  /**
   * If true, suppress all log messages from native CPP code
   */
  suppressNativeLog?: boolean;
  /**
   * Custom logger functions
   */
  logger?: WllamaLogger;
  /**
   * Maximum number of parallel files to be downloaded
   *
   * Default: parallelDownloads = 3
   */
  parallelDownloads?: number;
  /**
   * Allow offline mode. If true, the model will be loaded from cache if it's available.
   *
   * Default: allowOffline = false
   */
  allowOffline?: boolean;
  /**
   * Custom cache manager (only for advanced usage)
   */
  cacheManager?: CacheManager;
  /**
   * Custom model manager (only for advanced usage)
   */
  modelManager?: ModelManager;
}

export interface WllamaChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface AssetsPathConfig {
  default: string;
  'single-thread/wllama.wasm'?: string; // deprecated, use "default" instead
  'multi-thread/wllama.wasm'?: string; // deprecated, use "default" instead
}

export interface ModelMetadata {
  hparams: {
    nVocab: number;
    nCtxTrain: number;
    nEmbd: number;
    nLayer: number;
  };
  meta: Record<string, string>;
}

/**
 * Logger preset with debug messages suppressed
 */
export const LoggerWithoutDebug = {
  ...console,
  debug: () => {},
};

export type WllamaErrorType =
  | 'model_not_loaded'
  | 'download_error'
  | 'load_error'
  | 'kv_cache_full'
  | 'unknown_error'
  | 'inference_error';

export class WllamaError extends Error {
  type: WllamaErrorType;
  constructor(message: string, type: WllamaErrorType = 'unknown_error') {
    super(message);
    this.type = type;
  }
}

/**
 * AbortError is thrown when the user wants to abort the current operation.
 * This is equivalent to AbortError in Fetch API.
 */
export class WllamaAbortError extends Error {
  override name: string = 'AbortError';
  constructor() {
    super('Operation aborted');
  }
}

/**
 * Set compatibility options for Wllama.
 * By default, these are set to URL of the latest builds on CDN, which requires internet to download. If you want to use local assets or have your own CDN, follow the instruction from @wllama/wllama-compat package.
 */
export interface WllamaCompat {
  worker: string | { code: string };
  wasm: string;
}

export class Wllama {
  // The CacheManager and ModelManager are singleton, can be accessed by user
  public cacheManager: CacheManager;
  public modelManager: ModelManager;

  private compat: WllamaCompat | null = null;

  private proxy: ProxyToWorker = null as any;
  private config: WllamaConfig;
  private pathConfig: AssetsPathConfig;
  private useMultiThread: boolean = false;
  private nbThreads: number = 1;
  // private useEmbeddings: boolean = false;
  // available when loaded
  private loadedContextInfo: LoadedContextInfo = null as any;
  private seed: number | undefined = undefined;
  private bosToken: number = -1;
  private eosToken: number = -1;
  private eotToken: number = -1;
  private eogTokens: Set<number> = new Set();
  private addBosToken: boolean = false;
  private addEosToken: boolean = false;
  private mediaMarker?: string;
  private chatTemplate?: string;
  private metadata?: ModelMetadata;
  private hasEncoder: boolean = false;
  private decoderStartToken: number = -1;

  constructor(pathConfig: AssetsPathConfig, wllamaConfig: WllamaConfig = {}) {
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError('AssetsPathConfig is required');
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = wllamaConfig.cacheManager ?? new CacheManager();
    this.modelManager =
      wllamaConfig.modelManager ??
      new ModelManager({
        cacheManager: this.cacheManager,
        logger: wllamaConfig.logger ?? console,
        parallelDownloads: wllamaConfig.parallelDownloads,
        allowOffline: wllamaConfig.allowOffline,
      });
    this.setCompat('default');
  }

  private logger() {
    return this.config.logger ?? console;
  }

  private checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        'loadModel() is not yet called',
        'model_not_loaded'
      );
    }
  }

  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion(): string {
    return LIBLLAMA_VERSION;
  }

  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(
    compat: WllamaCompat | null | 'default',
    mode: 'safari' | 'firefox_safari' = 'safari'
  ) {
    if (mode === 'safari') {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === 'default' ? WasmCompatFromCDN : compat;
  }

  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded(): boolean {
    return !!this.proxy && !!this.metadata;
  }

  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS(): number {
    return this.bosToken;
  }

  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS(): number {
    return this.eosToken;
  }

  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT(): number {
    return this.eotToken;
  }

  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token: number): boolean {
    return (
      token === this.eosToken ||
      token === this.eotToken ||
      this.eogTokens.has(token)
    );
  }

  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken(): number {
    return this.decoderStartToken;
  }

  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata(): ModelMetadata {
    this.checkModelLoaded();
    return this.metadata!;
  }

  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread(): boolean {
    this.checkModelLoaded();
    return this.useMultiThread;
  }

  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads(): number {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }

  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture(): boolean {
    this.checkModelLoaded();
    return this.hasEncoder;
  }

  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken(): boolean {
    this.checkModelLoaded();
    return this.addBosToken;
  }

  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken(): boolean {
    this.checkModelLoaded();
    return this.addEosToken;
  }

  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate(): string | null {
    this.checkModelLoaded();
    return this.chatTemplate ?? null;
  }

  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU(): boolean {
    return isSupportWebGPU();
  }

  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  async loadModelFromUrl(
    modelSourceOrURL: ModelSource | string,
    params: LoadModelParams & DownloadOptions & { useCache?: boolean } = {}
  ): Promise<void> {
    const source: ModelSource = isString(modelSourceOrURL)
      ? ({ url: modelSourceOrURL } as ModelSource)
      : (modelSourceOrURL as ModelSource);
    const useCache = params.useCache ?? true;
    const model = useCache
      ? await this.modelManager.getModelOrDownload(source, params)
      : await this.modelManager.downloadModel(source, params);
    const blobs = await model.open();
    return await this.loadModel(blobs, params);
  }

  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  async loadModelFromHF(
    hfOptions: HuggingFaceParams,
    params: LoadModelParams & DownloadOptions & { useCache?: boolean } = {}
  ) {
    const source = await getHFModelSource(hfOptions);
    return await this.loadModelFromUrl(source, params);
  }

  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  async loadModel(
    ggufBlobsOrModel: Blob[] | Model,
    params: LoadModelParams = {}
  ): Promise<void> {
    const blobs: Blob[] =
      ggufBlobsOrModel instanceof Model
        ? await ggufBlobsOrModel.open()
        : [...(ggufBlobsOrModel as Blob[])]; // copy array
    if (blobs.some((b) => b.size === 0)) {
      throw new WllamaError(
        'Input model (or splits) must be non-empty Blob or File',
        'load_error'
      );
    }
    if (!this.pathConfig['default']) {
      throw new WllamaError(
        '"default" is missing from pathConfig',
        'load_error'
      );
    }

    if (this.proxy) {
      throw new WllamaError('Module is already initialized', 'load_error');
    }
    // detect if we can use multi-thread and webgpu
    const supportMultiThread = await isSupportMultiThread();
    const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
    const nbThreads = params.n_threads ?? hwConccurency;
    this.nbThreads = nbThreads;
    this.useMultiThread = supportMultiThread && nbThreads > 1;

    // prepare worker resources
    const workerResources: WllamaWorkerResources = {
      wasmPath: absoluteUrl(this.pathConfig['default']),
      compat: false,
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          'Not using compat mode' +
            (isFirefox()
              ? ' (expected on Firefox - WebGPU will be disabled)'
              : '')
        );
      } else {
        const isUsingDefault =
          this.compat.worker === WasmCompatFromCDN.worker &&
          this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            'Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package.'
          );
          this.logger().warn(
            'IMPORTANT: Performance will be significantly degraded in compatibility mode.'
          );
        }

        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }

    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }

    // initialize the worker
    this.proxy = new ProxyToWorker(
      workerResources,
      this.useMultiThread ? nbThreads : 0, // 0 means disable pthread
      this.config.suppressNativeLog ?? false,
      this.logger()
    );
    let logLevel = params.log_level ?? LogLevel.INFO;
    if (this.config.suppressNativeLog) {
      logLevel = 9999 as any;
    }

    const modelFiles = await prepareBlobs(blobs);
    await this.proxy.moduleInit(modelFiles.all);

    // run it
    this.logger().debug('Calling wllamaStart...');
    const startResult: any = await this.proxy.wllamaStart();
    if (!startResult.success) {
      throw new WllamaError(
        `Error while calling start function, result = ${startResult}`
      );
    }

    // load the model
    this.logger().debug('Loading model...');
    const loadResult: GlueMsgLoadRes = await this.proxy.wllamaAction('load', {
      _name: 'load_req',
      log_level: logLevel,
      // if async read is not supported, use mmap; refer to README-dev.md for more details
      use_mmap: !canUseAsyncFileRead(workerResources.compat),
      use_mlock: false,
      n_gpu_layers: params.n_gpu_layers ?? 99999,
      n_ctx: params.n_ctx ?? 1024,
      n_threads: this.useMultiThread ? nbThreads : 1,
      n_ctx_auto: false, // not supported for now
      mmproj_path: modelFiles.mmproj
        ? `/models/${MMPROJ_FILE_NAME}`
        : undefined,
      model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
      embeddings: params.embeddings,
      offload_kqv: params.offload_kqv,
      n_batch: params.n_batch,
      pooling_type: params.pooling_type as string,
      rope_scaling_type: params.rope_scaling_type as string,
      rope_freq_base: params.rope_freq_base,
      rope_freq_scale: params.rope_freq_scale,
      yarn_ext_factor: params.yarn_ext_factor,
      yarn_attn_factor: params.yarn_attn_factor,
      yarn_beta_fast: params.yarn_beta_fast,
      yarn_beta_slow: params.yarn_beta_slow,
      yarn_orig_ctx: params.yarn_orig_ctx,
      cache_type_k: params.cache_type_k as string,
      cache_type_v: params.cache_type_v as string,
      n_parallel: 1, // only support single sequence for now
      kv_unified: false, // TODO: support kv unified cache
      flash_attn: params.flash_attn,
      swa_full: params.swa_full,
      chat_template: params.chat_template,
      jinja: params.jinja,
    });
    const loadedCtxInfo: LoadedContextInfo & GlueMsgLoadRes = {
      ...loadResult,
      metadata: {},
    };
    for (let i = 0; i < loadResult.metadata_key.length; i++) {
      loadedCtxInfo.metadata[loadResult.metadata_key[i]] =
        loadResult.metadata_val[i];
    }
    this.seed = params.seed;
    this.bosToken = loadedCtxInfo.token_bos;
    this.eosToken = loadedCtxInfo.token_eos;
    this.eotToken = loadedCtxInfo.token_eot;
    // this.useEmbeddings = !!params.embeddings;
    this.metadata = {
      hparams: {
        nVocab: loadedCtxInfo.n_vocab,
        nCtxTrain: loadedCtxInfo.n_ctx_train,
        nEmbd: loadedCtxInfo.n_embd,
        nLayer: loadedCtxInfo.n_layer,
      },
      meta: loadedCtxInfo.metadata,
    };
    this.hasEncoder = !!loadedCtxInfo.has_encoder;
    this.decoderStartToken = loadedCtxInfo.token_decoder_start;
    this.addBosToken = loadedCtxInfo.add_bos_token;
    this.addEosToken = loadedCtxInfo.add_eos_token;
    this.chatTemplate = loadedCtxInfo.metadata['tokenizer.chat_template'];
    this.loadedContextInfo = loadedCtxInfo;
    this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
    this.mediaMarker = loadedCtxInfo.media_marker;
    this.logger().debug({ loadedCtxInfo });
  }

  getLoadedContextInfo(): LoadedContextInfo {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError('Loaded context info is not available');
    }
    // copy object
    return { ...this.loadedContextInfo };
  }

  //////////////////////////////////////////////
  // High level API

  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  async createEmbedding(
    options: EmbeddingCreateParams
  ): Promise<CreateEmbeddingResponse> {
    this.checkModelLoaded();

    const result = await this.proxy.wllamaAction<GlueMsgEmbeddingRes>(
      'embedding',
      {
        _name: 'embd_req',
        data_json: JSON.stringify(options),
        files: [], // TODO: support file input
      }
    );

    if (!result.success) {
      throw new WllamaError(
        'Model failed to start inference',
        'inference_error'
      );
    }

    return await this.getRespose(options as any, false);
  }

  /**
   * Make chat completion for a given chat messages.
   * @param options OAI-compatible chat completion options
   * @returns OAI-compatible chat completion response (only the final result when stream=false) or an async iterator of completion chunks (when stream=true)
   */
  async createChatCompletion(
    options: ChatCompletionParams & { stream?: false }
  ): Promise<ChatCompletionResponse>;
  async createChatCompletion(
    options: ChatCompletionParams & StreamParams<ChatCompletionChunk>
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  async createChatCompletion(
    options: ChatCompletionParams
  ): Promise<ChatCompletionResponse | AsyncIterable<ChatCompletionChunk>> {
    return options.stream
      ? await this.createCompletionGenerator(options)
      : await this.createCompletionImpl({ ...options, stream: false });
  }

  /**
   * Make (raw) completion for a given text.
   * @param options OAI-compatible completion options
   * @returns OAI-compatible completion response (only the final result when stream=false) or an async iterator of completion chunks (when stream=true)
   */
  async createCompletion(
    options: RawCompletionParams & { stream?: false }
  ): Promise<RawCompletionResponse>;
  async createCompletion(
    options: RawCompletionParams & StreamParams<RawCompletionChunk>
  ): Promise<AsyncIterable<RawCompletionChunk>>;
  async createCompletion(
    options: RawCompletionParams
  ): Promise<RawCompletionResponse | AsyncIterable<RawCompletionChunk>> {
    return options.stream
      ? await this.createCompletionGenerator(options)
      : await this.createCompletionImpl({ ...options, stream: false });
  }

  /**
   * Private implementation of createCompletion
   */
  private async createCompletionImpl<TOpt, TChunk>(
    options: TOpt
  ): Promise<TChunk> {
    this.checkModelLoaded();

    const isStream = !!(options as any).stream;
    const isChat = !!(options as any).messages;
    const customOpt: any = {};
    if (this.seed !== undefined) {
      customOpt.seed = this.seed;
    }
    let files: ArrayBuffer[] = [];
    if (isChat) {
      const tmp = this.prepareMultimodalInput(
        options as any as ChatCompletionParams
      );
      options = tmp.params as any;
      files = tmp.files;
    }
    const result = await this.proxy.wllamaAction<GlueMsgCompletionRes>(
      'completion',
      {
        _name: 'cmpl_req',
        is_chat: isChat,
        data_json: JSON.stringify({ ...options, ...customOpt }),
        files: files.map((f) => new Uint8Array(f)),
      }
    );

    if (!result.success) {
      throw new WllamaError(
        'Model failed to start inference',
        'inference_error'
      );
    }

    return await this.getRespose(options as StreamParams<TChunk>, isStream);
  }

  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   */
  private createCompletionGenerator<TOpt, TChunk>(
    options: Exclude<TOpt, 'onData'>
  ): Promise<AsyncIterable<TChunk>> {
    return new Promise((resolve, reject) => {
      const origOnData = (options as StreamParams<TChunk>).onData;
      const createGenerator = cbToAsyncIter(
        (callback: (val?: TChunk, done?: boolean) => void) => {
          this.createCompletionImpl<TOpt, TChunk>({
            ...options,
            onData: (chunk: TChunk) => {
              callback(chunk);
              origOnData?.(chunk);
            },
          })
            .catch(reject)
            .then(() => {
              callback(undefined, true);
            });
        }
      );
      resolve(createGenerator());
    });
  }

  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality: 'image' | 'audio'): boolean {
    this.checkModelLoaded();
    if (modality === 'image') {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === 'audio') {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        'Unsupported modality: ' + modality,
        'unknown_error'
      );
    }
  }

  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  async exit(): Promise<void> {
    await this.proxy?.wllamaExit();
    this.proxy = null as any;
  }

  //////////////////////////////////////////////
  // Low level API

  // TODO: add back

  /**
   * get debug info
   */
  async _getDebugInfo(): Promise<any> {
    this.checkModelLoaded();
    return await this.proxy.wllamaDebug();
  }

  //////////////////////////////////////////////
  // Utils

  private jsonDecode(data_json: string) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error('Failed to parse JSON:', data_json);
      throw new WllamaError('Failed to parse model output', 'inference_error');
    }
  }

  private prepareMultimodalInput(params: ChatCompletionParams): {
    params: ChatCompletionParams;
    files: ArrayBuffer[];
  } {
    const msg = params.messages;
    const msgNew: typeof msg = [];
    const files: ArrayBuffer[] = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent: typeof m.content = [];
        for (const c of m.content) {
          if (c.type === 'text') {
            // no transform for text content
            newContent.push(c);
          } else {
            // replace multimodal input with media marker
            if (!this.mediaMarker) {
              throw new WllamaError(
                'Media marker is undefined',
                'inference_error'
              );
            }
            files.push(c.data);
            newContent.push({
              type: 'text',
              text: this.mediaMarker,
            });
          }
        }
        msgNew.push({
          ...m,
          content: newContent,
        } as ChatCompletionUserMessage);
      } else {
        // no transform for non-typed content
        msgNew.push(m);
      }
    }
    return {
      params: {
        ...params,
        messages: msgNew,
      },
      files,
    };
  }

  private async getRespose(options: StreamParams<any>, isStream: boolean) {
    let finalResult: any = null;

    while (true) {
      if (options.abortSignal?.aborted) {
        throw new WllamaAbortError();
      }
      const result_chunk = await this.proxy.wllamaAction<GlueMsgGetResultRes>(
        'get_result',
        {
          _name: 'gres_req',
        }
      );

      const jsonString = result_chunk.data_json;
      if (!jsonString || jsonString.length === 0) {
        if (!result_chunk.has_more) {
          break;
        } else {
          continue;
        }
      }

      let jsonData = this.jsonDecode(jsonString);
      finalResult = jsonData;
      if (result_chunk.is_error) {
        this.logger().error('Model returned an error:', jsonData);
        throw new WllamaError(
          jsonData.message || 'Unknown inference error',
          'inference_error'
        );
      }

      if (isStream) {
        if (!Array.isArray(jsonData)) {
          jsonData = [jsonData];
        }

        for (const chunk of jsonData) {
          options.onData?.(chunk);
          finalResult = chunk;
        }
      }

      if (!result_chunk.has_more) {
        break;
      }
    }

    return finalResult;
  }
}
