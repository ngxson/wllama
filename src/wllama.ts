import { ProxyToWorker } from './worker';
import { absoluteUrl, bufToText, checkEnvironmentCompatible, isSupportMultiThread, joinBuffers, maybeSortFileByName, padDigits } from './utils';
import { CacheManager } from './cache-manager';
import { MultiDownloads } from './downloader/multi-downloads';

export interface WllamaConfig {
  /**
   * If true, suppress all log messages from native CPP code
   */
  suppressNativeLog?: boolean,
  /**
   * Custom logger functions
   */
  logger?: {
    debug: typeof console.debug,
    log: typeof console.log,
    warn: typeof console.warn,
    error: typeof console.error,
  },
};

export interface AssetsPathConfig {
  'single-thread/wllama.js': string,
  'single-thread/wllama.wasm': string,
  'multi-thread/wllama.js'?: string,
  'multi-thread/wllama.wasm'?: string,
  'multi-thread/wllama.worker.mjs'?: string,
};

export interface LoadModelConfig {
  seed?: number,
  n_ctx?: number,
  n_batch?: number,
  // by default, on multi-thread build, we take half number of available threads (hardwareConcurrency / 2)
  n_threads?: number,
  embeddings?: boolean,
  offload_kqv?: boolean,
  pooling_type?: 'LLAMA_POOLING_TYPE_UNSPECIFIED'
    | 'LLAMA_POOLING_TYPE_NONE'
    | 'LLAMA_POOLING_TYPE_MEAN'
    | 'LLAMA_POOLING_TYPE_CLS',
  // context extending
  rope_scaling_type?: 'LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED'
    | 'LLAMA_ROPE_SCALING_TYPE_NONE'
    | 'LLAMA_ROPE_SCALING_TYPE_LINEAR'
    | 'LLAMA_ROPE_SCALING_TYPE_YARN',
  rope_freq_base?: number,
  rope_freq_scale?: number,
  yarn_ext_factor?: number,
  yarn_attn_factor?: number,
  yarn_beta_fast?: number,
  yarn_beta_slow?: number,
  yarn_orig_ctx?: number,
  // TODO: add group attention
  // optimizations
  cache_type_k?: 'f16' | 'q8_0' | 'q4_0',
  cache_type_v?: 'f16',
};

export interface DownloadModelConfig extends LoadModelConfig {
  // download-specific params
  parallelDownloads?: number,
  progressCallback?: (opts: { loaded: number, total: number }) => any,
  /**
   * Default: useCache = true
   */
  useCache?: boolean,
};

export interface SamplingConfig {
  // See sampling.h for more details
  mirostat?: number, // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
  mirostat_tau?: number,
  temp?: number, // temperature
  top_p?: number,
  top_k?: number,
  penalty_last_n?: number,
  penalty_repeat?: number,
  penalty_freq?: number,
  penalty_present?: number,
  penalize_nl?: boolean,
  dynatemp_range?: number,
  dynatemp_exponent?: number,
  grammar?: string,
  n_prev?: number,
  n_probs?: number,
  min_p?: number,
  tfs_z?: number,
  typical_p?: number,
  logit_bias?: { token: number, bias: number }[],
};

export interface ChatCompletionOptions {
  nPredict?: number,
  onNewToken?(token: number, piece: Uint8Array, currentText: string, optionals: {
    abortSignal: () => any,
  }): any,
  sampling?: SamplingConfig,
  /**
   * List of custom token IDs for stopping the generation.  
   * Note: To convert from text to token ID, use lookupToken()
   */
  stopTokens?: number[],
};

export interface ModelMetadata {
  hparams: {
    nVocab: number;
    nCtxTrain: number;
    nEmbd: number;
    nLayer: number;
  },
  meta: Record<string, string>,
};

/**
 * Logger preset with debug messages suppressed
 */
export const LoggerWithoutDebug = {
  ...console,
  debug: () => {},
};

export class Wllama {
  // The CacheManager singleton, can be accessed by user
  public cacheManager = CacheManager;

  private proxy: ProxyToWorker = null as any;
  private config: WllamaConfig;
  private pathConfig: AssetsPathConfig;
  private useMultiThread: boolean = false;
  private useEmbeddings: boolean = false;
  // available when loaded
  private bosToken: number = -1;
  private eosToken: number = -1;
  private eotToken: number = -1;
  private metadata?: ModelMetadata;
  private samplingConfig: SamplingConfig = {};

  constructor(pathConfig: AssetsPathConfig, wllamaConfig: WllamaConfig = {}) {
    checkEnvironmentCompatible();
    if (!pathConfig) throw new Error('AssetsPathConfig is required');
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
  }

  private logger() {
    return this.config.logger ?? console;
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
   * Get model hyper-parameters and metadata
   * 
   * NOTE: This can only being used after `loadModel` is called.
   * 
   * @returns ModelMetadata
   */
  getModelMetadata(): ModelMetadata {
    if (!this.metadata) {
      throw new Error('loadModel() is not yet called');
    }
    return this.metadata;
  }

  /**
   * Check if we're currently using multi-thread build.
   * 
   * NOTE: This can only being used after `loadModel` is called.
   * 
   * @returns true if multi-thread is used.
   */
  isMultithread(): boolean {
    return this.useMultiThread;
  }

  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * If the input URL is an array, it returns the array itself.
   * If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  private parseModelUrl(modelUrl: string | string[]): string[] {
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /(?<baseURL>.*)-(?<current>\d{5})-of-(?<total>\d{5})\.gguf$/;
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches || !matches.groups || Object.keys(matches.groups).length !== 3) {
      return [modelUrl];
    }
    const { baseURL, total} = matches.groups
    const paddedShardIds = Array.from({ length: Number(total) }, (_, index) => (index + 1).toString().padStart(5, '0'));
    return paddedShardIds.map((current) => `${baseURL}-${current}-of-${total}.gguf`);
  }

  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * @param modelUrl URL or list of URLs (in the correct order)
   * @param config 
   */
  async loadModelFromUrl(modelUrl: string | string[], config: DownloadModelConfig = {}): Promise<void> {
    if (modelUrl.length === 0) {
      throw new Error('modelUrl must be an URL or a list of URLs (in the correct order)');
    }
    const skipCache = config.useCache === false;
    const multiDownloads = new MultiDownloads(
      this.logger(),
      this.parseModelUrl(modelUrl),
      config.parallelDownloads ?? 3,
      {
        progressCallback: config.progressCallback,
        useCache: !skipCache,
      }
    );
    const blobs = await multiDownloads.run();
    return await this.loadModel(blobs, config);
  }

  /**
   * Load model from a given list of Blob.
   * 
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   * 
   * @param ggufBlobs List of Blob that holds data of gguf file.
   * @param config LoadModelConfig
   */
  async loadModel(ggufBlobs: Blob[], config: LoadModelConfig = {}): Promise<void> {
    const blobs = [...ggufBlobs]; // copy array
    if (blobs.some(b => b.size === 0)) {
      throw new Error('Input model (or splits) must be non-empty Blob or File');
    }
    maybeSortFileByName(blobs);
    const hasMultipleBuffers = blobs.length > 1;
    if (this.proxy) {
      throw new Error('Module is already initialized');
    }
    // detect if we can use multi-thread
    const supportMultiThread = await isSupportMultiThread();
    if (!supportMultiThread) {
      this.logger().warn('Multi-threads are not supported in this environment, falling back to single-thread');
    }
    const hasPathMultiThread = !!this.pathConfig['multi-thread/wllama.js']
      && !!this.pathConfig['multi-thread/wllama.wasm']
      && !!this.pathConfig['multi-thread/wllama.worker.mjs'];
    if (!hasPathMultiThread) {
      this.logger().warn('Missing paths to "wllama.js", "wllama.wasm" or "wllama.worker.mjs", falling back to single-thread');
    }
    const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
    const nbThreads = config.n_threads ?? hwConccurency;
    this.useMultiThread = supportMultiThread && hasPathMultiThread && nbThreads > 1;
    const mPathConfig = this.useMultiThread
      ? {
        'wllama.js': absoluteUrl(this.pathConfig['multi-thread/wllama.js']!!),
        'wllama.wasm': absoluteUrl(this.pathConfig['multi-thread/wllama.wasm']!!),
        'wllama.worker.mjs': absoluteUrl(this.pathConfig['multi-thread/wllama.worker.mjs']!!),
      }
      : {
        'wllama.js': absoluteUrl(this.pathConfig['single-thread/wllama.js']),
        'wllama.wasm': absoluteUrl(this.pathConfig['single-thread/wllama.wasm']),
      };
    this.proxy = new ProxyToWorker(
      mPathConfig,
      this.useMultiThread ? nbThreads : 1,
      this.config.suppressNativeLog ?? false,
      this.logger(),
    );
    // TODO: files maybe out-of-order
    await this.proxy.moduleInit(blobs.map((blob, i) => ({
      name: hasMultipleBuffers
        ? `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
        : 'model.gguf',
      blob,
    })));
    // run it
    const startResult: any = await this.proxy.wllamaStart();
    if (!startResult.success) {
      throw new Error(`Error while calling start function, result = ${startResult}`);
    }
    // load the model
    const loadResult: {
      n_vocab: number,
      n_ctx_train: number,
      n_embd: number,
      n_layer: number,
      metadata: Record<string, string>,
      token_bos: number,
      token_eos: number,
      token_eot: number,
    } = await this.proxy.wllamaAction('load', {
      ...config,
      use_mmap: true,
      use_mlock: true,
      seed: config.seed || Math.floor(Math.random() * 100000),
      n_ctx: config.n_ctx || 1024,
      n_threads: this.useMultiThread ? nbThreads : 1,
      model_path: hasMultipleBuffers
        ? `/models/model-00001-of-${padDigits(blobs.length, 5)}.gguf`
        : '/models/model.gguf',
    });
    this.bosToken = loadResult.token_bos;
    this.eosToken = loadResult.token_eos;
    this.eotToken = loadResult.token_eot;
    this.useEmbeddings = !!config.embeddings;
    this.metadata = {
      hparams: {
        nVocab: loadResult.n_vocab,
        nCtxTrain: loadResult.n_ctx_train,
        nEmbd: loadResult.n_embd,
        nLayer: loadResult.n_layer,
      },
      meta: loadResult.metadata,
    };
  }

  //////////////////////////////////////////////
  // High level API

  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param text Input text
   * @returns An embedding vector
   */
  async createEmbedding(text: string, options: {
    skipBOS?: boolean,
    skipEOS?: boolean,
  } = {}): Promise<number[]> {
    const opt = {
      skipBOS: false,
      skipEOS: false,
      ...options,
    };
    if (!this.useEmbeddings) {
      throw new Error('embeddings is not enabled in LoadModelConfig')
    }
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
   * Make completion for a given text.
   * @param prompt Input text
   * @param options 
   * @returns Output completion text (only the completion part)
   */
  async createCompletion(prompt: string, options: ChatCompletionOptions): Promise<string> {
    this.samplingConfig = options.sampling ?? {};
    await this.samplingInit(this.samplingConfig);
    await this.kvClear(); // TODO: maybe cache tokens?
    const stopTokens = [
      this.eosToken,
      this.eotToken,
      ...(options.stopTokens ?? []),
    ]
    // process prompt
    const tokens = await this.tokenize(prompt, true);
    await this.samplingAccept(tokens);
    await this.decode(tokens, {});
    let outBuf = new Uint8Array();
    // abort signal
    let abort = false;
    const abortSignal = () => { abort = true };
    // predict next tokens
    for (let i = 0; i < (options.nPredict ?? Infinity); i++) {
      const sampled = await this.samplingSample();
      if (stopTokens.includes(sampled.token)) {
        break; // stop token
      }
      outBuf = joinBuffers([outBuf, sampled.piece]);
      if (options.onNewToken) {
        options.onNewToken(sampled.token, sampled.piece, bufToText(outBuf), {
          abortSignal,
        });
      }
      if (abort) {
        break; // abort signal is set
      }
      // decode next token
      await this.samplingAccept([sampled.token]);
      await this.decode([sampled.token], {});
    }
    return bufToText(outBuf);
  }

  //////////////////////////////////////////////
  // Low level API

  /**
   * Create or reset the ctx_sampling 
   * @param config 
   * @param pastTokens In case re-initializing the ctx_sampling, you can re-import past tokens into the new context
   */
  async samplingInit(config: SamplingConfig, pastTokens: number[] = []): Promise<void> {
    this.samplingConfig = config;
    const result = await this.proxy.wllamaAction('sampling_init', {
      ...config,
      tokens: pastTokens,
    });
    if (!result.success) {
      throw new Error('Failed to initialize sampling');
    }
  }

  /**
   * Get a list of pieces in vocab.  
   * NOTE: This function is slow, should only be used once.
   * @returns A list of Uint8Array. The nth element in the list associated to nth token in vocab
   */
  async getVocab(): Promise<Uint8Array[]> {
    const result = await this.proxy.wllamaAction('get_vocab', {});
    return result.vocab.map((arr: number[]) => new Uint8Array(arr));
  }

  /**
   * Lookup to see if a token exist in vocab or not. Useful for searching special tokens like "<|im_start|>"  
   * NOTE: It will match the whole token, so do not use it as a replacement for tokenize()  
   * @param piece 
   * @returns Token ID associated to the given piece. Returns -1 if cannot find the token.
   */
  async lookupToken(piece: string): Promise<number> {
    const result = await this.proxy.wllamaAction('lookup_token', { piece });
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
    const result = await this.proxy.wllamaAction('tokenize', special
      ? { text, special: true }
      : { text }
    );
    return result.tokens;
  }

  /**
   * Convert a list of tokens to text
   * @param tokens 
   * @returns Uint8Array, which maybe an unfinished unicode
   */
  async detokenize(tokens: number[]): Promise<Uint8Array> {
    const result = await this.proxy.wllamaAction('detokenize', { tokens });
    return new Uint8Array(result.buffer);
  }

  /**
   * Run llama_decode()
   * @param tokens A list of tokens to be decoded
   * @param options 
   * @returns n_past (number of tokens so far in the sequence)
   */
  async decode(tokens: number[], options: {
    skipLogits?: boolean,
  }): Promise<{ nPast: number }> {
    const req: any = { tokens };
    if (options.skipLogits) {
      req.skip_logits = true;
    }
    const result = await this.proxy.wllamaAction('decode', req);
    if (result.error) {
      throw new Error(result.error);
    } else if (!result.success) {
      throw new Error('Cannot decode, unknown error');
    } else {
      return { nPast: result.n_past };
    }
  }

  /**
   * Sample a new token (remember to samplingInit() at least once before calling this function)
   * @returns the token ID and its detokenized value (which maybe an unfinished unicode)
   */
  async samplingSample(): Promise<{ piece: Uint8Array, token: number }> {
    const result = await this.proxy.wllamaAction('sampling_sample', {});
    return {
      piece: new Uint8Array(result.piece),
      token: result.token,
    };
  }

  /**
   * Accept and save a new token to ctx_sampling
   * @param tokens 
   */
  async samplingAccept(tokens: number[]): Promise<void> {
    const result = await this.proxy.wllamaAction('sampling_accept', { tokens });
    if (!result.success) {
      throw new Error('samplingAccept unknown error');
    }
  }

  /**
   * Get softmax-ed probability of logits, can be used for custom sampling
   * @param topK Get top K tokens having highest logits value. If topK == -1, we return all n_vocab logits, but this is not recommended because it's slow.
   */
  async getLogits(topK: number = 40): Promise<{token: number, p: number}[]> {
    const result = await this.proxy.wllamaAction('get_logits', { top_k: topK });
    const logits = result.logits as number[][];
    return logits.map(([token, p]) => ({ token, p }));
  }

  /**
   * Calculate embeddings for a given list of tokens. Output vector is always normalized
   * @param tokens 
   * @returns A list of number represents an embedding vector of N dimensions
   */
  async embeddings(tokens: number[]): Promise<number[]> {
    const result = await this.proxy.wllamaAction('embeddings', { tokens });
    if (result.error) {
      throw new Error(result.error);
    } else if (!result.success) {
      throw new Error('embeddings unknown error');
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
    const result = await this.proxy.wllamaAction('kv_remove', {
      n_keep: nKeep,
      n_discard: nDiscard,
    });
    if (!result.success) {
      throw new Error('kvRemove unknown error');
    }
  }

  /**
   * Clear all tokens in KV cache
   */
  async kvClear(): Promise<void> {
    const result = await this.proxy.wllamaAction('kv_clear', {});
    if (!result.success) {
      throw new Error('kvClear unknown error');
    }
  }

  /**
   * Save session to file (virtual file system)  
   * TODO: add ability to download the file
   * @param filePath 
   * @returns List of tokens saved to the file
   */
  async sessionSave(filePath: string): Promise<{ tokens: number[] }> {
    const result = await this.proxy.wllamaAction('session_save', { session_path: filePath });
    return result;
  }

  /**
   * Load session from file (virtual file system)  
   * TODO: add ability to download the file
   * @param filePath 
   * 
   */
  async sessionLoad(filePath: string): Promise<void> {
    const result = await this.proxy.wllamaAction('session_load', { session_path: filePath });
    if (result.error) {
      throw new Error(result.error);
    } else if (!result.success) {
      throw new Error('sessionLoad unknown error');
    }
  }
  
  /**
   * Unload the model and free all memory
   */
  async exit(): Promise<void> {
    await this.proxy.wllamaExit();
  }

  /**
   * get debug info
   */
  async _getDebugInfo(): Promise<any> {
    return await this.proxy.wllamaDebug();
  }

  // TODO: add current_status
}