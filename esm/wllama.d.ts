export interface AssetsPathConfig {
    'single-thread/wllama.js': string;
    'single-thread/wllama.wasm': string;
    'multi-thread/wllama.js'?: string;
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
    pooling_type?: 'LLAMA_POOLING_TYPE_UNSPECIFIED' | 'LLAMA_POOLING_TYPE_NONE' | 'LLAMA_POOLING_TYPE_MEAN' | 'LLAMA_POOLING_TYPE_CLS';
    rope_scaling_type?: 'LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED' | 'LLAMA_ROPE_SCALING_TYPE_NONE' | 'LLAMA_ROPE_SCALING_TYPE_LINEAR' | 'LLAMA_ROPE_SCALING_TYPE_YARN';
    rope_freq_base?: number;
    rope_freq_scale?: number;
    yarn_ext_factor?: number;
    yarn_attn_factor?: number;
    yarn_beta_fast?: number;
    yarn_beta_slow?: number;
    yarn_orig_ctx?: number;
    cache_type_k?: 'f16' | 'q8_0' | 'q4_0';
    cache_type_v?: 'f16';
    n_download_parallel?: number;
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
    penalize_nl?: boolean;
    dynatemp_range?: number;
    dynatemp_exponent?: number;
    grammar?: string;
    n_prev?: number;
    n_probs?: number;
    min_p?: number;
    tfs_z?: number;
    typical_p?: number;
    logit_bias?: {
        token: number;
        bias: number;
    }[];
}
export declare class Wllama {
    private proxy;
    private pathConfig;
    private useMultiThread;
    private useEmbeddings;
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
    loadModel(ggufBuffer: Uint8Array | Uint8Array[], config: LoadModelConfig): Promise<void>;
    /**
     * Calculate embedding vector for a given text.
     * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
     * @param text Input text
     * @returns An embedding vector
     */
    createEmbedding(text: string, options?: {
        skipBOS?: boolean;
        skipEOS?: boolean;
    }): Promise<number[]>;
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
     * @param pastTokens In case re-initializing the ctx_sampling, you can re-import past tokens into the new context
     */
    samplingInit(config: SamplingConfig, pastTokens?: number[]): Promise<void>;
    /**
     * Get a list of pieces in vocab.
     * NOTE: This function is slow, should only be used once.
     * @returns A list of Uint8Array. The nth element in the list associated to nth token in vocab
     */
    getVocab(): Promise<Uint8Array[]>;
    /**
     * Lookup to see if a token exist in vocab or not. Useful for searching special tokens like "<|im_start|>"
     * NOTE: It will match the whole token, so do not use it as a replacement for tokenize()
     * @param piece
     * @returns Token ID associated to the given piece. Returns -1 if cannot find the token.
     */
    lookupToken(piece: string): Promise<number>;
    /**
     * Convert a given text to list of tokens
     * @param text
     * @param special Should split special tokens?
     * @returns List of token ID
     */
    tokenize(text: string, special?: boolean): Promise<number[]>;
    /**
     * Convert a list of tokens to text
     * @param tokens
     * @returns Uint8Array, which maybe an unfinished unicode
     */
    detokenize(tokens: number[]): Promise<Uint8Array>;
    /**
     * Run llama_decode()
     * @param tokens A list of tokens to be decoded
     * @param options
     * @returns n_past (number of tokens so far in the sequence)
     */
    decode(tokens: number[], options: {
        skipLogits?: boolean;
    }): Promise<{
        nPast: number;
    }>;
    /**
     * Sample a new token (remember to samplingInit() at least once before calling this function)
     * @returns the token ID and its detokenized value (which maybe an unfinished unicode)
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
     * Get softmax-ed probability of logits, can be used for custom sampling
     * @param topK Get top K tokens having highest logits value. If topK == -1, we return all n_vocab logits, but this is not recommended because it's slow.
     */
    getLogits(topK?: number): Promise<{
        token: number;
        p: number;
    }[]>;
    /**
     * Calculate embeddings for a given list of tokens. Output vector is always normalized
     * @param tokens
     * @returns A list of number represents an embedding vector of N dimensions
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
     * Save session to file (virtual file system)
     * TODO: add ability to download the file
     * @param filePath
     * @returns List of tokens saved to the file
     */
    sessionSave(filePath: string): Promise<{
        tokens: number[];
    }>;
    /**
     * Load session from file (virtual file system)
     * TODO: add ability to download the file
     * @param filePath
     *
     */
    sessionLoad(filePath: string): Promise<void>;
    /**
     * Unload the model and free all memory
     */
    exit(): Promise<void>;
}
