// Note: snake_case is used to match llama.cpp's naming convention
export interface LoadModelParams {
  log_level?: LogLevel;
  seed?: number;
  n_ctx?: number;
  n_batch?: number;
  // by default, all layers are offloaded if WebGPU is available
  n_gpu_layers?: number;
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
  // optimizations
  cache_type_k?: 'f32' | 'f16' | 'q8_0' | 'q5_1' | 'q5_0' | 'q4_1' | 'q4_0';
  cache_type_v?: 'f32' | 'f16' | 'q8_0' | 'q5_1' | 'q5_0' | 'q4_1' | 'q4_0';
  flash_attn?: boolean; // true is auto, false is disabled
  swa_full?: boolean;
  chat_template?: string;
  jinja?: boolean;
}

// Note: snake_case is used to match llama.cpp's naming convention
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
  has_image_input: boolean;
  has_audio_input: boolean;
}

// Note: snake_case is used to match llama.cpp's naming convention
export interface SamplingParams {
  // See sampling.h for more details
  seed?: number;
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

export interface StreamParams<T> {
  stream: true;
  onData: (data: T) => void;
  abortSignal?: AbortSignal;
}

export enum LogLevel {
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
}
