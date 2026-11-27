// Note: snake_case is used to match llama.cpp's naming convention
export interface LoadModelParams {
  log_level?: LogLevel;
  seed?: number;
  n_ctx?: number;
  n_batch?: number;
  n_ubatch?: number;
  // by default, all layers are offloaded if WebGPU is available
  n_gpu_layers?: number;
  // keep the token embedding on the GPU only, not also on the CPU, when all layers are on the GPU (default: true)
  // this saves its size in page memory; on a model with a separate output layer, it uses this size more GPU memory
  token_embd_on_gpu?: boolean;
  // MoE models: keep the experts on the CPU, same as --cpu-moe of llama.cpp (default: false). Not used with rpc workers
  moe_cpu?: boolean;
  // MoE models: GPU cache in MiB for the experts on the CPU, same as --moe-cache-mib of llama.cpp (default: 0, disabled)
  // the experts that a token uses are uploaded to the cache when they are not in it
  moe_cache_mib?: number;
  // number of extra wasm instances (rpc workers) that hold a part of the model, each one has its own 4GB memory
  // by default, they are used only when the part of the model on the CPU does not fit in one instance; set 0 to disable
  // WebGPU keeps the layers of n_gpu_layers, the workers get the other layers
  rpc_workers?: number;
  // by default, on multi-thread build, we take half number of available threads (hardwareConcurrency / 2)
  n_threads?: number;
  embeddings?: boolean;
  offload_kqv?: boolean;
  // maximum number of parallel requests (completions, embeddings, ...); default: 4
  n_parallel?: number;
  // share one KV cache of n_ctx tokens between all sequences (default: true)
  // when disabled, each sequence gets its own cache of (n_ctx / n_parallel) tokens
  kv_unified?: boolean;
  pooling_type?: // legacy values
  | 'LLAMA_POOLING_TYPE_UNSPECIFIED'
    | 'LLAMA_POOLING_TYPE_NONE'
    | 'LLAMA_POOLING_TYPE_MEAN'
    | 'LLAMA_POOLING_TYPE_CLS'
    // new values
    | 'unspecified'
    | 'none'
    | 'mean'
    | 'cls'
    | 'last'
    | 'rank';
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
  reasoning?: boolean;
  image_min_tokens?: number;
  image_max_tokens?: number;
  warmup?: boolean;
  no_kv_offload?: boolean;
  mmproj_offload?: boolean;
  cont_batching?: boolean;
  n_keep?: number;
  ctx_shift?: boolean;
  cache_idle_slots?: boolean;
  n_cache_reuse?: number;
  lora_adapters?: {
    path?: string; // path in the wasm file system
    blob?: Blob; // or the GGUF file of the adapter
    scale?: number;
  }[];
  lora_init_without_apply?: boolean;
  spec_draft_model?: string | Blob;
  spec_draft_ngl?: number;
  spec_draft_n_max?: number;
  spec_draft_n_min?: number;
  spec_draft_p_min?: number;
  spec_draft_threads?: number;
  spec_draft_threads_batch?: number;
  /** Comma-separated list, for example "ngram-simple" or "ngram-mod" */
  spec_type?: string;
  spec_ngram_simple_size_n?: number;
  spec_ngram_simple_size_m?: number;
  spec_ngram_simple_min_hits?: number;
  spec_ngram_mod_n_min?: number;
  spec_ngram_mod_n_max?: number;
  spec_ngram_mod_n_match?: number;
  kv_overrides?: Record<string, string>;
  reasoning_budget_tokens?: number;
  reasoning_budget_message?: string;
  reasoning_format?: 'none' | 'deepseek-legacy' | 'deepseek';
  skip_chat_parsing?: boolean;
  prefill_assistant?: boolean;
  default_template_kwargs?: Record<string, any>;
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
  has_video_input: boolean;
  has_audio_output: boolean;
  has_chat: boolean; // false for a TTS-only or a realtime transcription model
  has_audio_stream: boolean; // realtime sessions are available
  has_live_asr: boolean; // text comes out while the audio goes in, with intent: 'transcription'
  has_audio_duplex: boolean; // full duplex (e.g. Hibiki): a realtime session speaks while it listens, there are no turns
  audio_input_rate: number; // in Hz, 0 if there is no audio input
  audio_output_rate: number; // in Hz, 0 if there is no audio output
  audio_output_channels: number; // 1 = mono, 2 = stereo (pcm samples are interleaved), 0 if there is no audio output
  audio_voices: string[]; // predefined voices of createSpeech()
}

// Note: snake_case is used to match llama.cpp's naming convention
export interface SamplingParams {
  lora?: { id: number; scale: number }[]; // id: index of the LoRA adapter
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
  // token (ID or text) to bias, false bans the token
  logit_bias?:
    | Record<string, number | false>
    | [number | string, number | false][]
    | undefined;
  ignore_eos?: boolean | undefined;
}

export interface StreamParams<T> {
  stream: true;
  onData: (data: T) => void;
}

export enum LogLevel {
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
}
