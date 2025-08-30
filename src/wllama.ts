import { ProxyToWorker } from './worker';
import {
  absoluteUrl,
  bufToText,
  cbToAsyncIter,
  checkEnvironmentCompatible,
  isString,
  isSupportMultiThread,
  joinBuffers,
  sortFileByShard,
  isValidGgufFile,
} from './utils';
import CacheManager, { type DownloadOptions } from './cache-manager';
import { ModelManager, Model } from './model-manager';
import type {
  GlueMsgChatFormatRes,
  GlueMsgDecodeRes,
  GlueMsgDetokenizeRes,
  GlueMsgGetEmbeddingsRes,
  GlueMsgGetKvClearRes,
  GlueMsgGetKvRemoveRes,
  GlueMsgGetLogitsRes,
  GlueMsgGetVocabRes,
  GlueMsgLoadRes,
  GlueMsgLookupTokenRes,
  GlueMsgSamplingAcceptRes,
  GlueMsgSamplingSampleRes,
  GlueMsgSetOptionsRes,
  GlueMsgStatusRes,
  GlueMsgTestBenchmarkRes,
  GlueMsgTestPerplexityRes,
  GlueMsgTokenizeRes,
} from './glue/messages';
import { LIBLLAMA_VERSION } from './workers-code/generated';

const HF_MODEL_ID_REGEX = /^([a-zA-Z0-9_\-\.]+)\/([a-zA-Z0-9_\-\.]+)$/;
const HF_MODEL_ID_REGEX_EXPLAIN =
  "Hugging Face model ID is incorrect. Only regular alphanumeric characters, '-', '.' and '_' supported";

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
  'single-thread/wllama.wasm': string;
  'multi-thread/wllama.wasm'?: string;
}

export interface LoadModelConfig {
  seed?: number;
  n_ctx?: number;
  n_batch?: number;
  // by default, on multi-thread build, we take half number of available threads (hardwareConcurrency / 2)
  n_threads?: number;
  embeddings?: boolean;
  offload_kqv?: boolean;
  pooling_type?:
    | 'LLAMA_POOLING_TYPE_UNSPECIFIED'
    | 'LLAMA_POOLING_TYPE_NONE'
    | 'LLAMA_POOLING_TYPE_MEAN'
    | 'LLAMA_POOLING_TYPE_CLS';
  // context extending
  rope_scaling_type?:
    | 'LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED'
    | 'LLAMA_ROPE_SCALING_TYPE_NONE'
    | 'LLAMA_ROPE_SCALING_TYPE_LINEAR'
    | 'LLAMA_ROPE_SCALING_TYPE_YARN';
  rope_freq_base?: number;
  rope_freq_scale?: number;
  yarn_ext_factor?: number;
  yarn_attn_factor?: number;
  yarn_beta_fast?: number;
  yarn_beta_slow?: number;
  yarn_orig_ctx?: number;
  // TODO: add group attention
  // optimizations
  cache_type_k?: 'f32' | 'f16' | 'q8_0' | 'q5_1' | 'q5_0' | 'q4_1' | 'q4_0';
  cache_type_v?: 'f32' | 'f16' | 'q8_0' | 'q5_1' | 'q5_0' | 'q4_1' | 'q4_0';
  flash_attn?: boolean; // true is auto, false is disabled
}

export interface SamplingConfig {
  // See sampling.h for more details
  mirostat?: number | undefined; // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
  mirostat_eta?: number | undefined;
  mirostat_tau?: number | undefined;
  samplers_sequence?: string[] | undefined; // unused for now
  temp?: number | undefined; // temperature
  top_p?: number | undefined;
  top_k?: number | undefined;
  penalty_last_n?: number | undefined;
  penalty_repeat?: number | undefined;
  penalty_freq?: number | undefined;
  penalty_present?: number | undefined;
  dynatemp_range?: number | undefined;
  dynatemp_exponent?: number | undefined;
  grammar?: string;
  n_prev?: number | undefined;
  n_probs?: number | undefined;
  min_p?: number | undefined;
  typ_p?: number | undefined;
  typical_p?: number | undefined;
  logit_bias?: { token: number; bias: number }[] | undefined;
}

export interface CompletionChunk {
  token: number;
  piece: Uint8Array;
  currentText: string;
}

export interface CompletionOptions {
  /**
   * When processing input prompt, we don't need to get output tokens. Only used by llama_decode()
   * Default: false
   */
  skipLogits?: boolean;
  /**
   * Optional abort signal to stop the generation.
   * This can also be used to stop during prompt processing. In this case, it will throw WllamaAbortError.
   */
  abortSignal?: AbortSignal;
  /**
   * If true, return an AsyncIterable instead of a string
   */
  stream?: boolean;
}

export interface ChatCompletionOptions {
  nPredict?: number;
  onNewToken?(
    token: number,
    piece: Uint8Array,
    currentText: string,
    optionals: {
      /**
       * DEPRECATED, use ChatCompletionOptions["abortSignal"] instead
       */
      abortSignal: () => any;
    }
  ): any;
  sampling?: SamplingConfig;
  /**
   * List of custom token IDs for stopping the generation.
   * Note: To convert from text to token ID, use lookupToken()
   */
  stopTokens?: number[];
  /**
   * Equivalent to `cache_prompt` option in llama.cpp server.
   * Useful for chat, because it skip evaluating the history part of the conversation.
   */
  useCache?: boolean;
  /**
   * Optional abort signal to stop the generation.
   * This can also be used to stop during prompt processing (with a bit of delay.)
   */
  abortSignal?: AbortSignal;
  /**
   * If true, return an AsyncIterable instead of a string
   */
  stream?: boolean;
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

export interface ContextOptions {
  /**
   * Allow switching between embeddings / generation mode. Useful for models like GritLM.
   */
  embeddings: boolean;
}

export interface LoadedContextInfo {
  n_vocab: number;
  n_ctx: number;
  n_batch: number;
  n_ubatch: number;
  n_ctx_train: number;
  n_embd: number;
  n_layer: number;
  metadata: Record<string, string>;
  token_bos: number;
  token_eos: number;
  token_eot: number;
  list_tokens_eog: number[];
  has_encoder: boolean;
  token_decoder_start: number;
  add_bos_token: boolean;
  add_eos_token: boolean;
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

export class Wllama {
  // The CacheManager and ModelManager are singleton, can be accessed by user
  public cacheManager: CacheManager;
  public modelManager: ModelManager;

  private proxy: ProxyToWorker = null as any;
  private config: WllamaConfig;
  private pathConfig: AssetsPathConfig;
  private useMultiThread: boolean = false;
  private nbThreads: number = 1;
  private useEmbeddings: boolean = false;
  // available when loaded
  private loadedContextInfo: LoadedContextInfo = null as any;
  private bosToken: number = -1;
  private eosToken: number = -1;
  private eotToken: number = -1;
  private eogTokens: Set<number> = new Set();
  private addBosToken: boolean = false;
  private addEosToken: boolean = false;
  private chatTemplate?: string;
  private metadata?: ModelMetadata;
  private samplingConfig: SamplingConfig = {};
  private hasEncoder: boolean = false;
  private decoderStartToken: number = -1;
  private nCachedTokens: number = 0;

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
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelUrl URL to the GGUF file. If the model is splitted, pass the URL to the first shard.
   * @param config
   */
  async loadModelFromUrl(
    modelUrl: string | string[],
    config: LoadModelConfig & DownloadOptions & { useCache?: boolean } = {}
  ): Promise<void> {
    const url: string = isString(modelUrl) ? (modelUrl as string) : modelUrl[0];
    const useCache = config.useCache ?? true;
    const model = useCache
      ? await this.modelManager.getModelOrDownload(url, config)
      : await this.modelManager.downloadModel(url, config);
    const blobs = await model.open();
    return await this.loadModel(blobs, config);
  }

  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param modelId The HF model ID, for example: 'ggml-org/models'
   * @param filePath The GGUF file path, for example: 'tinyllamas/stories15M-q4_0.gguf'
   * @param config
   */
  async loadModelFromHF(
    modelId: string,
    filePath: string,
    config: LoadModelConfig & DownloadOptions & { useCache?: boolean } = {}
  ) {
    if (!modelId.match(HF_MODEL_ID_REGEX)) {
      throw new WllamaError(HF_MODEL_ID_REGEX_EXPLAIN, 'download_error');
    }
    if (!isValidGgufFile(filePath)) {
      throw new WllamaError('Only GGUF file is supported', 'download_error');
    }
    return await this.loadModelFromUrl(
      `https://huggingface.co/${modelId}/resolve/main/${filePath}`,
      config
    );
  }

  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param config LoadModelConfig
   */
  async loadModel(
    ggufBlobsOrModel: Blob[] | Model,
    config: LoadModelConfig = {}
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
    sortFileByShard(blobs);
    if (this.proxy) {
      throw new WllamaError('Module is already initialized', 'load_error');
    }
    // detect if we can use multi-thread
    const supportMultiThread = await isSupportMultiThread();
    if (!supportMultiThread) {
      this.logger().warn(
        'Multi-threads are not supported in this environment, falling back to single-thread'
      );
    }
    const hasPathMultiThread = !!this.pathConfig['multi-thread/wllama.wasm'];
    if (!hasPathMultiThread) {
      this.logger().warn(
        'Missing paths to "multi-thread/wllama.wasm", falling back to single-thread'
      );
    }
    const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
    const nbThreads = config.n_threads ?? hwConccurency;
    this.nbThreads = nbThreads;
    this.useMultiThread =
      supportMultiThread && hasPathMultiThread && nbThreads > 1;
    const mPathConfig = this.useMultiThread
      ? {
          'wllama.wasm': absoluteUrl(
            this.pathConfig['multi-thread/wllama.wasm']!!
          ),
        }
      : {
          'wllama.wasm': absoluteUrl(
            this.pathConfig['single-thread/wllama.wasm']
          ),
        };
    this.proxy = new ProxyToWorker(
      mPathConfig,
      this.useMultiThread ? nbThreads : 1,
      this.config.suppressNativeLog ?? false,
      this.logger()
    );
    const modelFiles = blobs.map((blob, i) => ({
      name: `model-${i}.gguf`,
      blob,
    }));
    await this.proxy.moduleInit(modelFiles);
    // run it
    const startResult: any = await this.proxy.wllamaStart();
    if (!startResult.success) {
      throw new WllamaError(
        `Error while calling start function, result = ${startResult}`
      );
    }
    // load the model
    const loadResult: GlueMsgLoadRes = await this.proxy.wllamaAction('load', {
      _name: 'load_req',
      use_mmap: true,
      use_mlock: true,
      n_gpu_layers: 0, // not supported for now
      seed: config.seed || Math.floor(Math.random() * 100000),
      n_ctx: config.n_ctx || 1024,
      n_threads: this.useMultiThread ? nbThreads : 1,
      n_ctx_auto: false, // not supported for now
      model_paths: modelFiles.map((f) => `models/${f.name}`),
      embeddings: config.embeddings,
      offload_kqv: config.offload_kqv,
      n_batch: config.n_batch,
      pooling_type: config.pooling_type as string,
      rope_scaling_type: config.rope_scaling_type as string,
      rope_freq_base: config.rope_freq_base,
      rope_freq_scale: config.rope_freq_scale,
      yarn_ext_factor: config.yarn_ext_factor,
      yarn_attn_factor: config.yarn_attn_factor,
      yarn_beta_fast: config.yarn_beta_fast,
      yarn_beta_slow: config.yarn_beta_slow,
      yarn_orig_ctx: config.yarn_orig_ctx,
      cache_type_k: config.cache_type_k as string,
      cache_type_v: config.cache_type_v as string,
      n_seq_max: 1, // only support single sequence for now
      flash_attn: config.flash_attn,
      swa_full: true, // TODO: properly support SWA
    });
    const loadedCtxInfo: LoadedContextInfo = {
      ...loadResult,
      metadata: {},
    };
    for (let i = 0; i < loadResult.metadata_key.length; i++) {
      loadedCtxInfo.metadata[loadResult.metadata_key[i]] =
        loadResult.metadata_val[i];
    }
    this.bosToken = loadedCtxInfo.token_bos;
    this.eosToken = loadedCtxInfo.token_eos;
    this.eotToken = loadedCtxInfo.token_eot;
    this.useEmbeddings = !!config.embeddings;
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
   * @param text Input text
   * @returns An embedding vector
   */
  async createEmbedding(
    text: string,
    options: {
      skipBOS?: boolean;
      skipEOS?: boolean;
    } = {}
  ): Promise<number[]> {
    this.checkModelLoaded();
    const opt = {
      skipBOS: false,
      skipEOS: false,
      ...options,
    };
    await this.samplingInit(this.samplingConfig);
    await this.kvClear();
    const tokens = await this.tokenize(text);
    if (this.bosToken && !opt.skipBOS) {
      tokens.unshift(this.bosToken);
    }
    if (this.eosToken && !opt.skipEOS) {
      tokens.push(this.eosToken);
    }
    const result = await this.embeddings(tokens);
    return result;
  }

  /**
   * Make completion for a given chat messages.
   *
   * NOTE: this function uses the chat template (if available) to format the chat messages. If the template is not available, it will use the default format (chatml). It can throw an error if the chat template is not compatible.
   *
   * @param messages Chat messages
   * @param options
   * @returns Output completion text (only the completion part)
   */
  async createChatCompletion(
    messages: WllamaChatMessage[],
    options: ChatCompletionOptions & { stream?: false }
  ): Promise<string>;
  async createChatCompletion(
    messages: WllamaChatMessage[],
    options: ChatCompletionOptions & { stream: true }
  ): Promise<AsyncIterable<CompletionChunk>>;
  async createChatCompletion(
    messages: WllamaChatMessage[],
    options: ChatCompletionOptions
  ): Promise<string | AsyncIterable<CompletionChunk>> {
    const prompt = await this.formatChat(messages, true);
    return options.stream
      ? await this.createCompletionGenerator(prompt, options)
      : await this.createCompletion(prompt, { ...options, stream: false });
  }

  /**
   * Make completion for a given text.
   * @param prompt Input text
   * @param options
   * @returns Output completion text (only the completion part)
   */
  async createCompletion(
    prompt: string,
    options: ChatCompletionOptions & { stream?: false }
  ): Promise<string>;
  async createCompletion(
    prompt: string,
    options: ChatCompletionOptions & { stream: true }
  ): Promise<AsyncIterable<CompletionChunk>>;
  async createCompletion(
    prompt: string,
    options: ChatCompletionOptions
  ): Promise<string | AsyncIterable<CompletionChunk>> {
    return options.stream
      ? await this.createCompletionGenerator(prompt, options)
      : await this.createCompletionImpl(prompt, { ...options, stream: false });
  }

  /**
   * Private implementation of createCompletion
   */
  private async createCompletionImpl(
    prompt: string,
    options: ChatCompletionOptions
  ): Promise<string> {
    this.checkModelLoaded();
    this.samplingConfig = options.sampling ?? {};
    await this.samplingInit(this.samplingConfig);
    const stopTokens = new Set(options.stopTokens ?? []);
    // process prompt
    let tokens = await this.tokenize(prompt, true);
    if (this.addBosToken && tokens[0] !== this.bosToken) {
      tokens.unshift(this.bosToken);
    }
    // maybe reuse KV cache
    if (options.useCache) {
      tokens = await this.computeNonCachedTokens(tokens);
    } else {
      await this.kvClear();
    }
    // decode/encode tokens
    await this.samplingAccept(tokens);
    if (this.isEncoderDecoderArchitecture()) {
      await this.encode(tokens);
      await this.decode([this.getDecoderStartToken()], {});
    } else {
      await this.decode(tokens, {});
    }
    let outBuf = new Uint8Array();
    // abort signal
    let abort = false;
    // abortSignalFn is a legacy function, use options.abortSignal instead
    const abortSignalFn = () => {
      abort = true;
    };
    // predict next tokens
    for (let i = 0; i < (options.nPredict ?? Infinity); i++) {
      const sampled = await this.samplingSample();
      if (this.isTokenEOG(sampled.token) || stopTokens.has(sampled.token)) {
        break; // stop token
      }
      // @ts-ignore Type 'Uint8Array<ArrayBufferLike>' is not assignable to type 'Uint8Array<ArrayBuffer>'
      outBuf = joinBuffers([outBuf, sampled.piece]);
      if (options.onNewToken) {
        options.onNewToken(sampled.token, sampled.piece, bufToText(outBuf), {
          abortSignal: abortSignalFn, // legacy
        });
      }
      if (abort || options.abortSignal?.aborted) {
        break; // abort signal is set
      }
      // decode next token
      await this.samplingAccept([sampled.token]);
      await this.decode([sampled.token], {});
    }
    return bufToText(outBuf);
  }

  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   */
  private createCompletionGenerator(
    prompt: string,
    options: Exclude<ChatCompletionOptions, 'onNewToken'>
  ): Promise<AsyncIterable<CompletionChunk>> {
    return new Promise((resolve, reject) => {
      const createGenerator = cbToAsyncIter(
        (callback: (val?: CompletionChunk, done?: boolean) => void) => {
          this.createCompletionImpl(prompt, {
            ...options,
            onNewToken: (token, piece, currentText) => {
              callback({ token, piece, currentText }, false);
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

  //////////////////////////////////////////////
  // Low level API

  /**
   * Create or reset the ctx_sampling
   * @param config
   * @param pastTokens In case re-initializing the ctx_sampling, you can re-import past tokens into the new context
   */
  async samplingInit(
    config: SamplingConfig,
    pastTokens: number[] = []
  ): Promise<void> {
    this.checkModelLoaded();
    this.samplingConfig = config;
    const logitBias = config.logit_bias ?? [];
    const logitBiasTok = logitBias.map((b) => b.token);
    const logitBiasVal = logitBias.map((b) => b.bias);
    const result = await this.proxy.wllamaAction<GlueMsgSamplingAcceptRes>(
      'sampling_init',
      {
        _name: 'sint_req',
        ...config,
        logit_bias_toks: logitBiasTok,
        logit_bias_vals: logitBiasVal,
        tokens: pastTokens,
      }
    );
    if (!result.success) {
      throw new WllamaError('Failed to initialize sampling');
    }
  }

  /**
   * Get a list of pieces in vocab.
   * NOTE: This function is slow, should only be used once.
   * @returns A list of Uint8Array. The nth element in the list associated to nth token in vocab
   */
  async getVocab(): Promise<Uint8Array[]> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgGetVocabRes>(
      'get_vocab',
      {
        _name: 'gvoc_req',
      }
    );
    return result.vocab;
  }

  /**
   * Lookup to see if a token exist in vocab or not. Useful for searching special tokens like "<|im_start|>"
   * NOTE: It will match the whole token, so do not use it as a replacement for tokenize()
   * @param piece
   * @returns Token ID associated to the given piece. Returns -1 if cannot find the token.
   */
  async lookupToken(piece: string): Promise<number> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgLookupTokenRes>(
      'lookup_token',
      {
        _name: 'lkup_req',
        piece,
      }
    );
    if (!result.success) {
      return -1;
    } else {
      return result.token as number;
    }
  }

  /**
   * Convert a given text to list of tokens
   * @param text
   * @param special Should split special tokens?
   * @returns List of token ID
   */
  async tokenize(text: string, special: boolean = true): Promise<number[]> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTokenizeRes>(
      'tokenize',
      {
        _name: 'tokn_req',
        text,
        special: !!special,
      }
    );
    return result.tokens;
  }

  /**
   * Convert a list of tokens to text
   * @param tokens
   * @param returnString Return a string instead of Uint8Array
   * @returns Uint8Array, which maybe an unfinished unicode
   */
  async detokenize(tokens: number[], returnString?: false): Promise<Uint8Array>;
  async detokenize(tokens: number[], returnString: true): Promise<string>;
  async detokenize(
    tokens: number[],
    returnString: true | false = false
  ): Promise<Uint8Array | string> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgDetokenizeRes>(
      'detokenize',
      {
        _name: 'dtkn_req',
        tokens,
      }
    );
    return returnString ? bufToText(result.buffer) : result.buffer;
  }

  /**
   * Run llama_decode()
   * @param tokens A list of tokens to be decoded
   * @param options Additional options
   * @returns n_past (number of tokens so far in the sequence)
   */
  async decode(
    tokens: number[],
    options: CompletionOptions
  ): Promise<{ nPast: number }> {
    this.checkModelLoaded();
    if (this.useEmbeddings) {
      throw new WllamaError(
        'embeddings is enabled. Use wllama.setOptions({ embeddings: false }) to disable it.'
      );
    }
    if (tokens.length === 0) {
      // do not call llama_decode if list of tokens is empty
      return {
        nPast: this.nCachedTokens,
      };
    }
    if (this.nCachedTokens + tokens.length > this.loadedContextInfo.n_ctx) {
      throw new WllamaError(
        'Running out of context cache. Please increase n_ctx when loading the model',
        'kv_cache_full'
      );
    }
    const batches = this.breakTokensIntoBatches(
      tokens,
      this.loadedContextInfo.n_batch
    );
    let result: any;
    for (let i = 0; i < batches.length; i++) {
      if (options?.abortSignal?.aborted) {
        throw new WllamaAbortError();
      }
      const isNotLast = batches.length > 1 && i < batches.length - 1;
      result = await this.proxy.wllamaAction<GlueMsgDecodeRes>('decode', {
        _name: 'deco_req',
        tokens: batches[i],
        skip_logits: options.skipLogits || isNotLast,
      });
      if (result.error) {
        throw new WllamaError(result.error);
      } else if (!result.success) {
        throw new WllamaError('Cannot encode, unknown error');
      }
    }
    this.nCachedTokens = result.n_past;
    return { nPast: result.n_past };
  }

  /**
   * Run llama_encode()
   * @param tokens A list of tokens to be encoded
   * @param options Additional options
   * @returns n_past (number of tokens so far in the sequence)
   */
  async encode(
    tokens: number[],
    options?: CompletionOptions
  ): Promise<{ nPast: number }> {
    this.checkModelLoaded();
    if (!this.hasEncoder) {
      throw new WllamaError(
        'This model does not use encoder-decoder architecture.',
        'inference_error'
      );
    }
    if (this.useEmbeddings) {
      throw new WllamaError(
        'embeddings is enabled. Use wllama.setOptions({ embeddings: false }) to disable it.',
        'inference_error'
      );
    }
    if (tokens.length === 0) {
      // do not call llama_encode if list of tokens is empty
      return {
        nPast: this.nCachedTokens,
      };
    }
    if (this.nCachedTokens + tokens.length > this.loadedContextInfo.n_ctx) {
      throw new WllamaError(
        'Running out of context cache. Please increase n_ctx when loading the model',
        'kv_cache_full'
      );
    }
    const batches = this.breakTokensIntoBatches(
      tokens,
      this.loadedContextInfo.n_batch
    );
    let result: any;
    for (let i = 0; i < batches.length; i++) {
      if (options?.abortSignal?.aborted) {
        throw new WllamaAbortError();
      }
      result = await this.proxy.wllamaAction<GlueMsgDecodeRes>('encode', {
        _name: 'enco_req',
        tokens: batches[i],
      });
      if (result.error) {
        throw new WllamaError(result.error);
      } else if (!result.success) {
        throw new WllamaError('Cannot encode, unknown error');
      }
    }
    this.nCachedTokens = result.n_past;
    return { nPast: result.n_past };
  }

  private breakTokensIntoBatches(
    tokens: number[],
    maxBatchSize: number
  ): number[][] {
    const batches: number[][] = [];
    for (let i = 0; i < tokens.length; i += maxBatchSize) {
      batches.push(tokens.slice(i, i + maxBatchSize));
    }
    return batches;
  }

  /**
   * Sample a new token (remember to samplingInit() at least once before calling this function)
   * @returns the token ID and its detokenized value (which maybe an unfinished unicode)
   */
  async samplingSample(): Promise<{ piece: Uint8Array; token: number }> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgSamplingSampleRes>(
      'sampling_sample',
      {
        _name: 'ssam_req',
      }
    );
    return {
      piece: result.piece,
      token: result.token,
    };
  }

  /**
   * Accept and save a new token to ctx_sampling
   * @param tokens
   */
  async samplingAccept(tokens: number[]): Promise<void> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgSamplingAcceptRes>(
      'sampling_accept',
      {
        _name: 'sacc_req',
        tokens,
      }
    );
    if (!result.success) {
      throw new WllamaError('samplingAccept unknown error');
    }
  }

  /**
   * Get softmax-ed probability of logits, can be used for custom sampling
   * @param topK Get top K tokens having highest logits value. If topK == -1, we return all n_vocab logits, but this is not recommended because it's slow.
   */
  async getLogits(topK: number = 40): Promise<{ token: number; p: number }[]> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgGetLogitsRes>(
      'get_logits',
      {
        _name: 'glog_req',
        top_k: topK,
      }
    );
    const logits: { token: number; p: number }[] = [];
    for (let i = 0; i < result.tokens.length; i++) {
      logits.push({
        token: result.tokens[i],
        p: result.probs[i],
      });
    }
    return logits;
  }

  /**
   * Calculate embeddings for a given list of tokens. Output vector is always normalized
   * @param tokens
   * @returns A list of number represents an embedding vector of N dimensions
   */
  async embeddings(tokens: number[]): Promise<number[]> {
    this.checkModelLoaded();
    if (!this.useEmbeddings) {
      throw new WllamaError(
        'embeddings is disabled. Use wllama.setOptions({ embeddings: true }) to enable it.',
        'inference_error'
      );
    }
    if (this.nCachedTokens > 0) {
      this.logger().warn(
        'Embeddings: KV cache is not empty, this may produce incorrect results'
      );
    }
    if (this.nCachedTokens + tokens.length > this.loadedContextInfo.n_ctx) {
      throw new WllamaError(
        'Running out of context cache. Please increase n_ctx when loading the model',
        'kv_cache_full'
      );
    }
    if (tokens.length > this.loadedContextInfo.n_batch) {
      throw new WllamaError(
        'Embedding tokens does not fit into batch. Please increase n_batch when loading the model',
        'inference_error'
      );
    }
    if (tokens.length > this.loadedContextInfo.n_ubatch) {
      throw new WllamaError(
        'Embedding tokens does not fit into physical batch. Please increase n_ubatch when loading the model',
        'inference_error'
      );
    }
    const result = await this.proxy.wllamaAction<GlueMsgGetEmbeddingsRes>(
      'embeddings',
      {
        _name: 'gemb_req',
        tokens,
      }
    );
    if (!result.success) {
      throw new WllamaError('embeddings unknown error');
    } else {
      return result.embeddings;
    }
  }

  /**
   * Remove and shift some tokens from KV cache.
   * Keep n_keep, remove n_discard then shift the rest
   * @param nKeep
   * @param nDiscard
   */
  async kvRemove(nKeep: number, nDiscard: number): Promise<void> {
    this.checkModelLoaded();
    if (nDiscard === 0) return;
    const result = await this.proxy.wllamaAction<GlueMsgGetKvRemoveRes>(
      'kv_remove',
      {
        _name: 'kvcr_req',
        n_keep: nKeep,
        n_discard: nDiscard,
      }
    );
    if (!result.success) {
      throw new WllamaError('kvRemove unknown error');
    }
    // When nDiscard is negative (-1), it means remove everything after nKeep
    if (nDiscard < 0) {
      this.nCachedTokens = nKeep;
    } else {
      this.nCachedTokens -= nDiscard;
    }
  }

  /**
   * Clear all tokens in KV cache
   */
  async kvClear(): Promise<void> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgGetKvClearRes>(
      'kv_clear',
      {
        _name: 'kvcc_req',
      }
    );
    if (!result.success) {
      throw new WllamaError('kvClear unknown error');
    }
    this.nCachedTokens = 0;
  }

  /**
   * Save session to file (virtual file system)
   * TODO: add ability to download the file
   * @param filePath
   * @returns List of tokens saved to the file
   */
  // async sessionSave(filePath: string): Promise<{ tokens: number[] }> {
  //   this.checkModelLoaded();
  //   const result = await this.proxy.wllamaAction('session_save', {
  //     session_path: filePath,
  //   });
  //   return result;
  // }

  /**
   * Load session from file (virtual file system)
   * TODO: add ability to download the file
   * @param filePath
   */
  // async sessionLoad(filePath: string): Promise<void> {
  //   this.checkModelLoaded();
  //   const result = await this.proxy.wllamaAction('session_load', {
  //     session_path: filePath,
  //   });
  //   if (result.error) {
  //     throw new WllamaError(result.error);
  //   } else if (!result.success) {
  //     throw new WllamaError('sessionLoad unknown error');
  //   }
  //   const cachedTokens = await this.getCachedTokens();
  //   this.nCachedTokens = cachedTokens.length;
  // }

  /**
   * Apply chat template to a list of messages
   *
   * @param messages list of messages
   * @param addAssistant whether to add assistant prompt at the end
   * @param template (optional) custom template, see llama-server --chat-template argument for more details
   * @returns formatted chat
   */
  async formatChat(
    messages: WllamaChatMessage[],
    addAssistant: boolean,
    template?: string
  ): Promise<string> {
    this.checkModelLoaded();
    const roles = messages.map((m) => m.role);
    const contents = messages.map((m) => m.content);
    const result = await this.proxy.wllamaAction<GlueMsgChatFormatRes>(
      'chat_format',
      {
        _name: 'cfmt_req',
        roles,
        contents,
        tmpl: template,
        add_ass: addAssistant,
      }
    );
    if (!result.success) {
      throw new WllamaError('formatChat unknown error');
    }
    return result.formatted_chat;
  }

  /**
   * Set options for underlaying llama_context
   */
  async setOptions(opt: ContextOptions): Promise<void> {
    this.checkModelLoaded();
    await this.proxy.wllamaAction<GlueMsgSetOptionsRes>('set_options', {
      _name: 'opti_req',
      ...opt,
    });
    this.useEmbeddings = opt.embeddings;
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

  /**
   * get debug info
   */
  async _getDebugInfo(): Promise<any> {
    this.checkModelLoaded();
    return await this.proxy.wllamaDebug();
  }

  /**
   * benchmark function, only used internally
   */
  async _testBenchmark(
    type: 'tg' | 'pp',
    nSamples: number
  ): Promise<{ t_ms: number }> {
    this.checkModelLoaded();
    return await this.proxy.wllamaAction<GlueMsgTestBenchmarkRes>(
      'test_benchmark',
      {
        _name: 'tben_req',
        type,
        n_samples: nSamples,
      }
    );
  }

  /**
   * perplexity function, only used internally
   */
  async _testPerplexity(tokens: number[]): Promise<{ ppl: number }> {
    this.checkModelLoaded();
    return await this.proxy.wllamaAction<GlueMsgTestPerplexityRes>(
      'test_perplexity',
      {
        _name: 'tper_req',
        tokens,
      }
    );
  }

  ///// Prompt cache utils /////
  private async getCachedTokens(): Promise<number[]> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgStatusRes>(
      'current_status',
      {
        _name: 'stat_req',
      }
    );
    return result.tokens;
  }

  /**
   * Compare the input sequence and cachedToken, then return the part that is not in cache.
   * This function also remove mismatch part in cache (via kvRemove)
   */
  private async computeNonCachedTokens(seq: number[]): Promise<number[]> {
    const cachedTokens = await this.getCachedTokens();
    let nKeep = 0;
    for (; nKeep < Math.min(cachedTokens.length, seq.length); nKeep++) {
      if (cachedTokens[nKeep] !== seq[nKeep]) {
        break;
      }
    }
    this.logger().debug(`Cache nKeep=${nKeep}`);
    try {
      await this.kvRemove(nKeep, -1);
      return seq.slice(nKeep, seq.length);
    } catch (e) {
      this.logger().warn('Failed to rollback KV cache, clearing it instead');
      await this.kvClear();
      return seq;
    }
  }

  // TODO: add current_status
}
