import type { SamplingParams } from './types';

// Message content types

export interface ChatCompletionMessageText {
  type: 'text';
  text: string;
}

export interface ChatCompletionMessageImage {
  type: 'image';
  data: ArrayBuffer;
}

export interface ChatCompletionMessageAudio {
  type: 'audio';
  data: ArrayBuffer;
}

export type ChatCompletionMessageContent =
  | ChatCompletionMessageText
  | ChatCompletionMessageImage
  | ChatCompletionMessageAudio;

// Tool definitions

export interface ChatCompletionToolFunctionParameters {
  type: 'object';
  properties: Record<
    string,
    {
      type: string;
      description?: string;
      enum?: string[];
      [key: string]: unknown;
    }
  >;
  required?: string[];
  additionalProperties?: boolean;
}

export interface ChatCompletionToolFunction {
  name: string;
  description?: string;
  parameters?: ChatCompletionToolFunctionParameters;
  strict?: boolean;
}

export interface ChatCompletionTool {
  type: 'function';
  function: ChatCompletionToolFunction;
}

export type ChatCompletionToolChoice =
  | 'none'
  | 'auto'
  | 'required'
  | { type: 'function'; function: { name: string } };

// Message types

export interface ChatCompletionSystemMessage {
  role: 'system';
  content: string;
  name?: string;
}

export interface ChatCompletionUserMessage {
  role: 'user';
  content: string | ChatCompletionMessageContent[];
  name?: string;
}

export interface ChatCompletionToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string; // JSON-encoded string
  };
}

export interface ChatCompletionAssistantMessage {
  role: 'assistant';
  content?: string | null;
  name?: string;
  tool_calls?: ChatCompletionToolCall[];
}

export interface ChatCompletionToolMessage {
  role: 'tool';
  content: string;
  tool_call_id: string;
}

export type ChatCompletionMessage =
  | ChatCompletionSystemMessage
  | ChatCompletionUserMessage
  | ChatCompletionAssistantMessage
  | ChatCompletionToolMessage;

// Request params

export type ChatCompletionParams = {
  messages: ChatCompletionMessage[];
  stream?: boolean;
  model?: string;
  // sampling
  temperature?: number;
  max_tokens?: number;
  // stop?: string | string[];
  // n?: number;
  // logprobs?: boolean;
  // top_logprobs?: number;
  // logit_bias?: Record<string, number>;
  // tools
  tools?: ChatCompletionTool[];
  tool_choice?: ChatCompletionToolChoice;
  // parallel_tool_calls?: boolean;
  // response format
  response_format?: {
    type: 'text' | 'json_object' | 'json_schema';
    json_schema?: { name: string; schema: unknown; strict?: boolean };
  };
  // user-facing
  user?: string;
} & SamplingParams;

// Response types----------

export interface ChatCompletionLogprob {
  token: string;
  logprob: number;
  bytes: number[] | null;
}

export interface ChatCompletionLogprobsContent extends ChatCompletionLogprob {
  top_logprobs: ChatCompletionLogprob[];
}

export interface ChatCompletionChoiceLogprobs {
  content: ChatCompletionLogprobsContent[] | null;
  refusal: ChatCompletionLogprobsContent[] | null;
}

export interface ChatCompletionChoice {
  index: number;
  message: ChatCompletionAssistantMessage;
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
  logprobs: ChatCompletionChoiceLogprobs | null;
}

export interface ChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: { cached_tokens: number; audio_tokens: number };
  completion_tokens_details?: {
    reasoning_tokens: number;
    audio_tokens: number;
    accepted_prediction_tokens: number;
    rejected_prediction_tokens: number;
  };
}

/** Response when stream=false (or omitted) */
export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: ChatCompletionUsage;
  system_fingerprint?: string;
  service_tier?: string;
}

// Streaming response types

export interface ChatCompletionChunkDelta {
  role?: 'assistant';
  content?: string | null;
  tool_calls?: Array<{
    index: number;
    id?: string;
    type?: 'function';
    function?: { name?: string; arguments?: string };
  }>;
  refusal?: string | null;
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: ChatCompletionChunkDelta;
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
  logprobs: ChatCompletionChoiceLogprobs | null;
}

export interface ResultTimings {
  cache_n: number;
  prompt_n: number;
  prompt_ms: number;
  prompt_per_token_ms: number;
  prompt_per_second: number;
  predicted_n: number;
  predicted_ms: number;
  predicted_per_token_ms: number;
  predicted_per_second: number;
}

/** Response when stream=true — one chunk per SSE event */
export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
  usage?: ChatCompletionUsage | null;
  timings?: ResultTimings;
}

// Raw (text) completion

export type RawCompletionParams = {
  prompt: string | string[];
  stream?: boolean;
  model?: string;
  suffix?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  logprobs?: number | null;
  echo?: boolean;
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  best_of?: number;
  logit_bias?: Record<string, number>;
  seed?: number;
  user?: string;
} & SamplingParams;

export interface RawCompletionChoice {
  text: string;
  index: number;
  finish_reason: 'stop' | 'length' | 'content_filter' | null;
  logprobs: {
    tokens: string[];
    token_logprobs: number[];
    top_logprobs: Array<Record<string, number>>;
    text_offset: number[];
  } | null;
}

/** Response when stream=false */
export interface RawCompletionResponse {
  id: string;
  object: 'text_completion';
  created: number;
  model: string;
  choices: RawCompletionChoice[];
  usage: ChatCompletionUsage;
  system_fingerprint?: string;
  timings?: ResultTimings;
}

/** One chunk when stream=true */
export interface RawCompletionChunk {
  id: string;
  object: 'text_completion';
  created: number;
  model: string;
  choices: Array<{
    text: string;
    index: number;
    finish_reason: 'stop' | 'length' | 'content_filter' | null;
    logprobs: null;
  }>;
  usage?: ChatCompletionUsage | null;
  timings?: ResultTimings;
}

// Embeddings

export interface EmbeddingCreateParams {
  input: string | string[] | number[] | number[][];
  model?: string;
  encoding_format?: 'float' | 'base64';
  // dimensions?: number; // unsupported by llama.cpp
  // user?: string;
}

export interface Embedding {
  object: 'embedding';
  index: number;
  embedding: number[] | string; // float array or base64 string depending on encoding_format
}

export interface EmbeddingUsage {
  prompt_tokens: number;
  total_tokens: number;
}

export interface CreateEmbeddingResponse {
  object: 'list';
  data: Embedding[];
  model: string;
  usage: EmbeddingUsage;
}
