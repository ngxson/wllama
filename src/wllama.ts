import ModuleSingleThread from './single-thread/wllama';
import ModuleMultiThread from './multi-thread/wllama';
import { bufToText, delay, getWModuleConfig, isSupportMultiThread, joinBuffers, loadBinaryResource } from './utils';

export interface AssetsPathConfig {
  'single-thread/wllama.wasm': string,
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
  // context extending
  rope_scaling_type?: number,
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
  grammar?: string,
  n_prev?: number,
  n_probs?: number,
  min_p?: number,
  tfs_z?: number,
  typical_p?: number,
};

export class Wllama {
  private wModule?: any;
  private pathConfig: AssetsPathConfig;
  private useMultiThread: boolean = false;
  private wllamaStart: any;
  private wllamaAction: any = () => {throw new Error('Model is not yet loaded')};
  private wllamaExit: any = () => {throw new Error('Model is not yet loaded')};
  // available when loaded
  private bosToken: number = -1;
  private eosToken: number = -1;
  private samplingConfig: SamplingConfig = {};

  constructor(config: AssetsPathConfig) {
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

  private _callWrapper(name: string, ret: string, args: string[]) {
    const fn = this.wModule.cwrap(name, ret, args);
    const decodeException = this.wModule.cwrap('wllama_decode_exception', 'string', ['number']);
    return async (action: string, req: any): Promise<any> => {
      let result: any;
      try {
        if (args.length === 2) {
          result = JSON.parse(await fn(action, JSON.stringify(req)));
        } else {
          result = fn();
        }
      } catch (ex) {
        const what = await decodeException(ex);
        console.error(what);
        throw new Error(what);
      }
      return result;
    };
  }

  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * @param modelUrl URL or list of URLs (in the correct order)
   * @param config 
   */
  async loadModelFromUrl(modelUrl: string | string[], config: LoadModelConfig): Promise<void> {
    const ggufBuffer = await loadBinaryResource(modelUrl);
    return await this.loadModel(ggufBuffer, config);
  }

  /**
   * Load model from a given buffer
   * @param ggufBuffer Uint8Array holds data of gguf file
   * @param config 
   */
  async loadModel(ggufBuffer: Uint8Array, config: LoadModelConfig): Promise<void> {
    if (!ggufBuffer.byteLength) {
      throw new Error('Input model must be a non-empty Uint8Array');
    }
    if (this.wModule) {
      throw new Error('Module is already initialized');
    }
    // detect if we can use multi-thread
    const supportMultiThread = await isSupportMultiThread();
    if (!supportMultiThread) {
      console.warn('Multi-threads are not supported in this environment, falling back to single-thread');
    }
    const hasPathMultiThread = !!this.pathConfig['multi-thread/wllama.wasm'] && !!this.pathConfig['multi-thread/wllama.worker.mjs'];
    if (!hasPathMultiThread) {
      console.warn('Missing paths to "wllama.wasm" and "wllama.worker.mjs", falling back to single-thread');
    }
    const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
    const nbThreads = config.n_threads ?? hwConccurency;
    this.useMultiThread = supportMultiThread && hasPathMultiThread && nbThreads > 1;
    this.wModule = this.useMultiThread
      ? await ModuleMultiThread(getWModuleConfig({
        'wllama.wasm': this.pathConfig['multi-thread/wllama.wasm']!!,
        'wllama.worker.mjs': this.pathConfig['multi-thread/wllama.worker.mjs']!!,
      }))
      : await ModuleSingleThread(getWModuleConfig({
        'wllama.wasm': this.pathConfig['single-thread/wllama.wasm'],
      }));
    // create vfs folder for storing model bins
    this.wModule['FS_createPath']('/', 'models', true, true);
    // load model
    this.wModule['FS_createDataFile']('/models', 'model.bin', ggufBuffer, true, true, true);
    // initialize bindings
    this.wllamaStart = this._callWrapper('wllama_start', 'number', []);
    this.wllamaAction = this._callWrapper('wllama_action', 'string', ['string', 'string']);
    this.wllamaExit = this._callWrapper('wllama_exit', 'number', []);
    // run it
    const startResult: number = await this.wllamaStart();
    if (startResult !== 0) {
      throw new Error(`Error while calling start function, result = ${startResult}`);
    }
    // load the model
    const loadResult: {
      token_bos: number,
      token_eos: number,
    } = await this.wllamaAction('load', {
      ...config,
      seed: config.seed || Math.floor(Math.random() * 100000),
      n_ctx: config.n_ctx || 1024,
      n_threads: this.useMultiThread ? nbThreads : 1,
      model_path: '/models/model.bin',
    });
    this.bosToken = loadResult.token_bos;
    this.eosToken = loadResult.token_eos;
  }

  //////////////////////////////////////////////
  // High level API

  /**
   * Calculate embedding vector for a given text
   * @param text Input text
   * @returns An embedding vector
   */
  async createEmbeddings(text: string): Promise<number[]> {
    await this.samplingInit(this.samplingConfig);
    await this.kvClear();
    const tokens = await this.tokenize(text);
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
      if (!this.useMultiThread) {
        // if this is single-thread, we add a setTimeout to allow frontend code to run
        // TODO: we should somehow use web worker
        await delay(0);
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
   */
  async samplingInit(config: {
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
    grammar?: string,
    n_prev?: number,
    n_probs?: number,
    min_p?: number,
    tfs_z?: number,
    typical_p?: number,
  }): Promise<void> {
    this.samplingConfig = config;
    const result = await this.wllamaAction('sampling_init', config);
    if (!result.success) {
      throw new Error('Failed to initialize sampling');
    }
  }

  /**
   * Lookup to see if a token exist in vocab or not. Useful for searching special tokens like "<|im_start|>"
   * NOTE: It will match the whole token, so do not use it as a replacement for tokenize()
   * @param piece 
   * @returns 
   */
  async lookupToken(piece: string) {
    const result = await this.wllamaAction('lookup_token', { piece });
    if (!result.success) {
      return -1;
    } else {
      return result.token;
    }
  }

  /**
   * Convert a given text to list of tokens
   * @param text 
   * @param special Should split special tokens?
   * @returns 
   */
  async tokenize(text: string, special: boolean = true): Promise<number[]> {
    const result = await this.wllamaAction('tokenize', special
      ? { text, special: true }
      : { text }
    );
    return result.tokens;
  }

  /**
   * Convert a list of tokens to text
   * @param tokens 
   * @returns 
   */
  async detokenize(tokens: number[]): Promise<Uint8Array> {
    const result = await this.wllamaAction('detokenize', { tokens });
    return new Uint8Array(result.buffer);
  }

  /**
   * Run llama_decode()
   * @param tokens A list of tokens to be decoded
   * @param options 
   * @returns n_past
   */
  async decode(tokens: number[], options: {
    skipLogits?: boolean,
  }): Promise<{ nPast: number }> {
    const req: any = { tokens };
    if (options.skipLogits) {
      req.skip_logits = true;
    }
    const result = await this.wllamaAction('decode', req);
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
   * @returns 
   */
  async samplingSample(): Promise<{ piece: Uint8Array, token: number }> {
    const result = await this.wllamaAction('sampling_sample', {});
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
    const result = await this.wllamaAction('sampling_accept', { tokens });
    if (!result.success) {
      throw new Error('samplingAccept unknown error');
    }
  }

  /**
   * Calculate embeddings for a given list of tokens
   * @param tokens 
   * @returns 
   */
  async embeddings(tokens: number[]): Promise<number[]> {
    const result = await this.wllamaAction('embeddings', { tokens });
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
    const result = await this.wllamaAction('kv_remove', {
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
    const result = await this.wllamaAction('kv_clear', {});
    if (!result.success) {
      throw new Error('kvClear unknown error');
    }
  }
  
  /**
   * Unload the model and free all memory
   */
  async exit(): Promise<void> {
    await this.wllamaExit();
  }

  // TODO: add session save / session load / current_status
}