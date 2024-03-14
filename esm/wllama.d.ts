export interface AssetsPathConfig {
    'single-thread/wllama.wasm': string;
    'multi-thread/wllama.wasm'?: string;
    'multi-thread/wllama.worker.mjs'?: string;
}
export interface LoadModelConfig {
    seed?: number;
    n_ctx?: number;
    n_batch?: number;
    n_threads?: number;
    embeddings?: boolean;
    offload_kqv?: boolean;
    n_seq_max?: number;
    rope_scaling_type?: number;
    rope_freq_base?: number;
    rope_freq_scale?: number;
    yarn_ext_factor?: number;
    yarn_attn_factor?: number;
    yarn_beta_fast?: number;
    yarn_beta_slow?: number;
    yarn_orig_ctx?: number;
    cache_type_k?: 'f16' | 'q8_0' | 'q4_0';
    cache_type_v?: 'f16';
}
export interface SamplingConfig {
    mirostat?: number;
    mirostat_tau?: number;
    temp?: number;
    top_p?: number;
    top_k?: number;
    penalty_last_n?: number;
    penalty_repeat?: number;
    penalty_freq?: number;
    penalty_present?: number;
    grammar?: string;
    n_prev?: number;
    n_probs?: number;
    min_p?: number;
    tfs_z?: number;
    typical_p?: number;
}
export declare class Wllama {
    private wModule?;
    private pathConfig;
    private useMultiThread;
    private wllamaStart;
    private wllamaAction;
    private wllamaExit;
    private bosToken;
    private eosToken;
    private samplingConfig;
    constructor(config: AssetsPathConfig);
    /**
     * Get token ID associated to BOS (begin of sentence) token
     * @returns -1 if the model is not loaded.
     */
    getBOS(): number;
    /**
     * Get token ID associated to EOS (end of sentence) token
     * @returns -1 if the model is not loaded.
     */
    getEOS(): number;
    /**
     * Check if we're currently using multi-thread build
     * @returns true if multi-thread is used
     */
    isMultithread(): boolean;
    private _callWrapper;
    /**
     * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
     * @param modelUrl URL or list of URLs (in the correct order)
     * @param config
     */
    loadModelFromUrl(modelUrl: string | string[], config: LoadModelConfig): Promise<void>;
    /**
     * Load model from a given buffer
     * @param ggufBuffer Uint8Array holds data of gguf file
     * @param config
     */
    loadModel(ggufBuffer: Uint8Array, config: LoadModelConfig): Promise<void>;
    /**
     * Calculate embedding vector for a given text
     * @param text Input text
     * @returns An embedding vector
     */
    createEmbeddings(text: string): Promise<number[]>;
    /**
     * Make completion for a given text
     * @param prompt Input text
     * @param options
     * @returns Output completion text (only the completion part)
     */
    createCompletion(prompt: string, options: {
        nPredict?: number;
        onNewToken?(token: number, piece: Uint8Array, currentText: string): any;
        sampling?: SamplingConfig;
    }): Promise<string>;
    /**
     * Create or reset the ctx_sampling
     * @param config
     */
    samplingInit(config: {
        mirostat?: number;
        mirostat_tau?: number;
        temp?: number;
        top_p?: number;
        top_k?: number;
        penalty_last_n?: number;
        penalty_repeat?: number;
        penalty_freq?: number;
        penalty_present?: number;
        grammar?: string;
        n_prev?: number;
        n_probs?: number;
        min_p?: number;
        tfs_z?: number;
        typical_p?: number;
    }): Promise<void>;
    /**
     * Lookup to see if a token exist in vocab or not. Useful for searching special tokens like "<|im_start|>"
     * NOTE: It will match the whole token, so do not use it as a replacement for tokenize()
     * @param piece
     * @returns
     */
    lookupToken(piece: string): Promise<any>;
    /**
     * Convert a given text to list of tokens
     * @param text
     * @param special Should split special tokens?
     * @returns
     */
    tokenize(text: string, special?: boolean): Promise<number[]>;
    /**
     * Convert a list of tokens to text
     * @param tokens
     * @returns
     */
    detokenize(tokens: number[]): Promise<Uint8Array>;
    /**
     * Run llama_decode()
     * @param tokens A list of tokens to be decoded
     * @param options
     * @returns n_past
     */
    decode(tokens: number[], options: {
        skipLogits?: boolean;
    }): Promise<{
        nPast: number;
    }>;
    /**
     * Sample a new token (remember to samplingInit() at least once before calling this function)
     * @returns
     */
    samplingSample(): Promise<{
        piece: Uint8Array;
        token: number;
    }>;
    /**
     * Accept and save a new token to ctx_sampling
     * @param tokens
     */
    samplingAccept(tokens: number[]): Promise<void>;
    /**
     * Calculate embeddings for a given list of tokens
     * @param tokens
     * @returns
     */
    embeddings(tokens: number[]): Promise<number[]>;
    /**
     * Remove and shift some tokens from KV cache.
     * Keep n_keep, remove n_discard then shift the rest
     * @param nKeep
     * @param nDiscard
     */
    kvRemove(nKeep: number, nDiscard: number): Promise<void>;
    /**
     * Clear all tokens in KV cache
     */
    kvClear(): Promise<void>;
    /**
     * Unload the model and free all memory
     */
    exit(): Promise<void>;
}
