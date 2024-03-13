import ModuleSingleThread from './single-thread/wllama';
import ModuleMultiThread from './multi-thread/wllama';
import { bufToText, delay, getWModuleConfig, isSupportMultiThread, joinBuffers, loadBinaryResource } from './utils';

export interface AssetsPathConfig {
  wasmSingleThreadPath: string,
  wasmMultiThreadPath?: string,
  workerMultiThreadPath?: string,
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
  wModule?: any;
  pathConfig: AssetsPathConfig;
  _wllamaStart: any;
  _wllamaAction: any = () => {throw new Error('Model is not yet loaded')};
  _wllamaExit: any = () => {throw new Error('Model is not yet loaded')};
  // available when loaded
  _bosToken: number = -1;
  _eosToken: number = -1;
  _samplingConfig: SamplingConfig = {};

  constructor(config: AssetsPathConfig) {
    if (!config) throw new Error('AssetsPathConfig is required');
    this.pathConfig = config;
  }

  _callWrapper(name: string, ret: string, args: string[]) {
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

  async loadModel(modelUrl: string | string[], config: LoadModelConfig): Promise<void> {
    if (this.wModule) {
      throw new Error('Module is already initialized');
    }
    // detect if we can use multi-thread
    const supportMultiThread = await isSupportMultiThread();
    if (!supportMultiThread) {
      console.warn('Multi-threads are supported in this environment, falling back to single-thread');
    }
    const hasPathMultiThread = !!this.pathConfig.wasmMultiThreadPath && !!this.pathConfig.workerMultiThreadPath;
    if (!hasPathMultiThread) {
      console.warn('Missing paths to "wllama.wasm" and "wllama.worker.mjs", falling back to single-thread');
    }
    const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
    const nbThreads = config.n_threads ?? hwConccurency;
    const useMultiThread = supportMultiThread && hasPathMultiThread && nbThreads > 1;
    this.wModule = useMultiThread
      ? await ModuleMultiThread(getWModuleConfig({
        'wllama.wasm': this.pathConfig.wasmMultiThreadPath!!,
        'wllama.worker.mjs': this.pathConfig.workerMultiThreadPath!!,
      }))
      : await ModuleSingleThread(getWModuleConfig({
        'wllama.wasm': this.pathConfig.wasmSingleThreadPath,
      }));
    const modelData = await loadBinaryResource(modelUrl);
    // create vfs folder for storing model bins
    this.wModule['FS_createPath']('/', 'models', true, true);
    // load model
    this.wModule['FS_createDataFile']('/models', 'model.bin', modelData, true, true, true);
    // initialize bindings
    this._wllamaStart = this._callWrapper('wllama_start', 'number', []);
    this._wllamaAction = this._callWrapper('wllama_action', 'string', ['string', 'string']);
    this._wllamaExit = this._callWrapper('wllama_exit', 'number', []);
    // run it
    const startResult: number = await this._wllamaStart();
    if (startResult !== 0) {
      throw new Error(`Error while calling start function, result = ${startResult}`);
    }
    // load the model
    const loadResult: {
      token_bos: number,
      token_eos: number,
    } = await this._wllamaAction('load', {
      ...config,
      seed: config.seed || Math.floor(Math.random() * 100000),
      n_ctx: config.n_ctx || 1024,
      n_threads: useMultiThread ? nbThreads : 1,
      model_path: '/models/model.bin',
    });
    this._bosToken = loadResult.token_bos;
    this._eosToken = loadResult.token_eos;
  }

  //////////////////////////////////////////////
  // High level API

  async createEmbeddings(text: string): Promise<number[]> {
    await this.samplingInit(this._samplingConfig);
    await this.kvClear();
    const tokens = await this.tokenize(text);
    const result = await this.embeddings(tokens);
    return result;
  }

  async createCompletion(prompt: string, options: {
    nPredict?: number,
    onNewToken?(token: number, piece: Uint8Array, currentText: string): any,
    sampling?: SamplingConfig,
  }): Promise<string> {
    this._samplingConfig = options.sampling ?? {};
    await this.samplingInit(this._samplingConfig);
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
      if (sampled.token === this._eosToken) {
        break; // EOS token
      }
      outBuf = joinBuffers([outBuf, sampled.piece]);
      if (options.onNewToken) {
        options.onNewToken(sampled.token, sampled.piece, bufToText(outBuf));
      }
      await delay(1); // time to update frontend
      // decode next token
      await this.samplingAccept([sampled.token]);
      await this.decode([sampled.token], {});
    }
    return bufToText(outBuf);
  }

  //////////////////////////////////////////////
  // Low level API

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
    this._samplingConfig = config;
    const result = await this._wllamaAction('sampling_init', config);
    if (!result.success) {
      throw new Error('Failed to initialize sampling');
    }
  }

  async lookupToken(piece: string) {
    const result = await this._wllamaAction('lookup_token', { piece });
    if (!result.success) {
      return -1;
    } else {
      return result.token;
    }
  }

  async tokenize(text: string, special: boolean = true): Promise<number[]> {
    const result = await this._wllamaAction('tokenize', special
      ? { text, special: true }
      : { text }
    );
    return result.tokens;
  }

  async detokenize(tokens: number[]): Promise<Uint8Array> {
    const result = await this._wllamaAction('detokenize', { tokens });
    return new Uint8Array(result.buffer);
  }

  async decode(tokens: number[], options: {
    skipLogits?: boolean,
  }): Promise<{ nPast: number }> {
    const req: any = { tokens };
    if (options.skipLogits) {
      req.skip_logits = true;
    }
    const result = await this._wllamaAction('decode', req);
    if (result.error) {
      throw new Error(result.error);
    } else if (!result.success) {
      throw new Error('Cannot decode, unknown error');
    } else {
      return { nPast: result.n_past };
    }
  }

  async samplingSample(): Promise<{ piece: Uint8Array, token: number }> {
    const result = await this._wllamaAction('sampling_sample', {});
    return {
      piece: new Uint8Array(result.piece),
      token: result.token,
    };
  }

  async samplingAccept(tokens: number[]): Promise<void> {
    const result = await this._wllamaAction('sampling_accept', { tokens });
    if (!result.success) {
      throw new Error('samplingAccept unknown error');
    }
  }

  async embeddings(tokens: number[]): Promise<number[]> {
    const result = await this._wllamaAction('embeddings', { tokens });
    if (result.error) {
      throw new Error(result.error);
    } else if (!result.success) {
      throw new Error('embeddings unknown error');
    } else {
      return result.embeddings;
    }
  }

  async kvRemove(nKeep: number, nDiscard: number): Promise<void> {
    const result = await this._wllamaAction('kv_remove', {
      n_keep: nKeep,
      n_discard: nDiscard,
    });
    if (!result.success) {
      throw new Error('kvRemove unknown error');
    }
  }

  async kvClear(): Promise<void> {
    const result = await this._wllamaAction('kv_clear', {});
    if (!result.success) {
      throw new Error('kvClear unknown error');
    }
  }

  // TODO: add session save / session load / current_status
}