export interface AssetsPathConfig {
    wasmSingleThreadPath: string;
    wasmMultiThreadPath?: string;
    workerMultiThreadPath?: string;
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
    wModule?: any;
    pathConfig: AssetsPathConfig;
    _wllamaStart: any;
    _wllamaAction: any;
    _wllamaExit: any;
    _bosToken: number;
    _eosToken: number;
    _samplingConfig: SamplingConfig;
    constructor(config: AssetsPathConfig);
    _callWrapper(name: string, ret: string, args: string[]): (action: string, req: any) => Promise<any>;
    loadModel(modelUrl: string | string[], config: LoadModelConfig): Promise<void>;
    createEmbeddings(text: string): Promise<number[]>;
    createCompletion(prompt: string, options: {
        nPredict?: number;
        onNewToken?(token: number, piece: Uint8Array, currentText: string): any;
        sampling?: SamplingConfig;
    }): Promise<string>;
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
    lookupToken(piece: string): Promise<any>;
    tokenize(text: string, special?: boolean): Promise<number[]>;
    detokenize(tokens: number[]): Promise<Uint8Array>;
    decode(tokens: number[], options: {
        skipLogits?: boolean;
    }): Promise<{
        nPast: number;
    }>;
    samplingSample(): Promise<{
        piece: Uint8Array;
        token: number;
    }>;
    samplingAccept(tokens: number[]): Promise<void>;
    embeddings(tokens: number[]): Promise<number[]>;
    kvRemove(nKeep: number, nDiscard: number): Promise<void>;
    kvClear(): Promise<void>;
}
