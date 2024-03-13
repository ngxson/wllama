import ModuleSingleThread from './single-thread/wllama.js';
import ModuleMultiThread from './multi-thread/wllama.js';
import { bufToText, delay, getWModuleConfig, isSupportMultiThread, joinBuffers, loadBinaryResource } from './utils.js';
;
;
;
export class Wllama {
    wModule;
    pathConfig;
    _wllamaStart;
    _wllamaAction = () => { throw new Error('Model is not yet loaded'); };
    _wllamaExit = () => { throw new Error('Model is not yet loaded'); };
    // available when loaded
    _bosToken = -1;
    _eosToken = -1;
    _samplingConfig = {};
    constructor(config) {
        if (!config)
            throw new Error('AssetsPathConfig is required');
        this.pathConfig = config;
    }
    _callWrapper(name, ret, args) {
        const fn = this.wModule.cwrap(name, ret, args);
        const decodeException = this.wModule.cwrap('wllama_decode_exception', 'string', ['number']);
        return async (action, req) => {
            let result;
            try {
                if (args.length === 2) {
                    result = JSON.parse(await fn(action, JSON.stringify(req)));
                }
                else {
                    result = fn();
                }
            }
            catch (ex) {
                const what = await decodeException(ex);
                console.error(what);
                throw new Error(what);
            }
            return result;
        };
    }
    async loadModel(modelUrl, config) {
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
                'wllama.wasm': this.pathConfig.wasmMultiThreadPath,
                'wllama.worker.mjs': this.pathConfig.workerMultiThreadPath,
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
        const startResult = await this._wllamaStart();
        if (startResult !== 0) {
            throw new Error(`Error while calling start function, result = ${startResult}`);
        }
        // load the model
        const loadResult = await this._wllamaAction('load', {
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
    async createEmbeddings(text) {
        await this.samplingInit(this._samplingConfig);
        await this.kvClear();
        const tokens = await this.tokenize(text);
        const result = await this.embeddings(tokens);
        return result;
    }
    async createCompletion(prompt, options) {
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
    async samplingInit(config) {
        this._samplingConfig = config;
        const result = await this._wllamaAction('sampling_init', config);
        if (!result.success) {
            throw new Error('Failed to initialize sampling');
        }
    }
    async lookupToken(piece) {
        const result = await this._wllamaAction('lookup_token', { piece });
        if (!result.success) {
            return -1;
        }
        else {
            return result.token;
        }
    }
    async tokenize(text, special = true) {
        const result = await this._wllamaAction('tokenize', special
            ? { text, special: true }
            : { text });
        return result.tokens;
    }
    async detokenize(tokens) {
        const result = await this._wllamaAction('detokenize', { tokens });
        return new Uint8Array(result.buffer);
    }
    async decode(tokens, options) {
        const req = { tokens };
        if (options.skipLogits) {
            req.skip_logits = true;
        }
        const result = await this._wllamaAction('decode', req);
        if (result.error) {
            throw new Error(result.error);
        }
        else if (!result.success) {
            throw new Error('Cannot decode, unknown error');
        }
        else {
            return { nPast: result.n_past };
        }
    }
    async samplingSample() {
        const result = await this._wllamaAction('sampling_sample', {});
        return {
            piece: new Uint8Array(result.piece),
            token: result.token,
        };
    }
    async samplingAccept(tokens) {
        const result = await this._wllamaAction('sampling_accept', { tokens });
        if (!result.success) {
            throw new Error('samplingAccept unknown error');
        }
    }
    async embeddings(tokens) {
        const result = await this._wllamaAction('embeddings', { tokens });
        if (result.error) {
            throw new Error(result.error);
        }
        else if (!result.success) {
            throw new Error('embeddings unknown error');
        }
        else {
            return result.embeddings;
        }
    }
    async kvRemove(nKeep, nDiscard) {
        const result = await this._wllamaAction('kv_remove', {
            n_keep: nKeep,
            n_discard: nDiscard,
        });
        if (!result.success) {
            throw new Error('kvRemove unknown error');
        }
    }
    async kvClear() {
        const result = await this._wllamaAction('kv_clear', {});
        if (!result.success) {
            throw new Error('kvClear unknown error');
        }
    }
}
//# sourceMappingURL=wllama.js.map