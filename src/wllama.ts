import { ProxyToWorker } from './worker';
import { absoluteUrl, bufToText, checkEnvironmentCompatible, isSupportMultiThread, joinBuffers, loadBinaryResource, padDigits } from './utils';

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
  n_threads?: number,
  embeddings?: boolean,
  offload_kqv?: boolean,
  n_seq_max?: number,
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
  // download-specific params
  n_download_parallel?: number,
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

export class Wllama {
  private proxy: ProxyToWorker = null as any;
  private pathConfig: AssetsPathConfig;
  private useMultiThread: boolean = false;
  private useEmbeddings: boolean = false;
  // available when loaded
  private bosToken: number = -1;
  private eosToken: number = -1;
  private samplingConfig: SamplingConfig = {};

  constructor(config: AssetsPathConfig) {
    checkEnvironmentCompatible();
    if (!config) throw new Error('AssetsPathConfig is required');
    this.pathConfig = config;
  }

  /**
   * Get token ID associated to BOS (begin of sentence) token
   * @returns -1 if the model is not loaded.
   */
  getBOS(): number {
    return this.bosToken;
  }

  /**
   * Get token ID associated to EOS (end of sentence) token
   * @returns -1 if the model is not loaded.
   */
  getEOS(): number {
    return this.bosToken;
  }

  /**
   * Check if we're currently using multi-thread build
   * @returns true if multi-thread is used
   */
  isMultithread(): boolean {
    return this.useMultiThread;
  }

  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * @param modelUrl URL or list of URLs (in the correct order)
   * @param config 
   */
  async loadModelFromUrl(modelUrl: string | string[], config: LoadModelConfig): Promise<void> {
    if (modelUrl.length === 0) {
      throw new Error('modelUrl must be an URL or a list of URLs (in the correct order)');
    }
    const ggufBuffers = await loadBinaryResource(modelUrl, config.n_download_parallel ?? 3);
    return await this.loadModel(ggufBuffers, config);
  }

  /**
   * Load model from a given buffer
   * @param ggufBuffer Uint8Array holds data of gguf file
   * @param config 
   */
  async loadModel(ggufBuffer: Uint8Array | Uint8Array[], config: LoadModelConfig): Promise<void> {
    const buffers: Uint8Array[] = Array.isArray(ggufBuffer)
      ? ggufBuffer as Uint8Array[]
      : [ggufBuffer as Uint8Array];
    if (buffers.length === 0 || buffers.some(buf => buf.byteLength === 0)) {
      throw new Error('Input model (or splits) must be non-empty Uint8Array');
    }
    if (this.proxy) {
      throw new Error('Module is already initialized');
    }
    // detect if we can use multi-thread
    const supportMultiThread = await isSupportMultiThread();
    if (!supportMultiThread) {
      console.warn('Multi-threads are not supported in this environment, falling back to single-thread');
    }
    const hasPathMultiThread = !!this.pathConfig['multi-thread/wllama.js']
      && !!this.pathConfig['multi-thread/wllama.wasm']
      && !!this.pathConfig['multi-thread/wllama.worker.mjs'];
    if (!hasPathMultiThread) {
      console.warn('Missing paths to "wllama.js", "wllama.wasm" or "wllama.worker.mjs", falling back to single-thread');
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
    this.proxy = new ProxyToWorker(mPathConfig, this.useMultiThread);
    await this.proxy.moduleInit(buffers);
    // run it
    const startResult: number = await this.proxy.wllamaStart();
    if (startResult !== 0) {
      throw new Error(`Error while calling start function, result = ${startResult}`);
    }
    // load the model
    const loadResult: {
      token_bos: number,
      token_eos: number,
    } = await this.proxy.wllamaAction('load', {
      ...config,
      seed: config.seed || Math.floor(Math.random() * 100000),
      n_ctx: config.n_ctx || 1024,
      n_threads: this.useMultiThread ? nbThreads : 1,
      model_path: buffers.length > 1
        ? `/models/model-00001-of-${padDigits(buffers.length, 5)}.gguf`
        : '/models/model.gguf',
    });
    this.bosToken = loadResult.token_bos;
    this.eosToken = loadResult.token_eos;
    this.useEmbeddings = !!config.embeddings;
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
   * Make completion for a given text
   * @param prompt Input text
   * @param options 
   * @returns Output completion text (only the completion part)
   */
  async createCompletion(prompt: string, options: {
    nPredict?: number,
    onNewToken?(token: number, piece: Uint8Array, currentText: string): any,
    sampling?: SamplingConfig,
  }): Promise<string> {
    this.samplingConfig = options.sampling ?? {};
    await this.samplingInit(this.samplingConfig);
    await this.kvClear(); // TODO: maybe cache tokens?
    // process prompt
    const tokens = await this.tokenize(prompt, true);
    await this.samplingAccept(tokens);
    await this.decode(tokens, {});
    let outBuf = new Uint8Array();
    // predict next tokens
    for (let i = 0; i < (options.nPredict ?? Infinity); i++) {
      const sampled = await this.samplingSample();
      // TODO: add support stop sequence
      if (sampled.token === this.eosToken) {
        break; // EOS token
      }
      outBuf = joinBuffers([outBuf, sampled.piece]);
      if (options.onNewToken) {
        options.onNewToken(sampled.token, sampled.piece, bufToText(outBuf));
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

  // TODO: add current_status
}