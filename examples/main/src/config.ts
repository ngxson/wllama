// See: https://vitejs.dev/guide/assets#explicit-url-imports
import wllamaSingleJS from '@wllama/wllama/src/single-thread/wllama.js?url';
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMultiJS from '@wllama/wllama/src/multi-thread/wllama.js?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';
import wllamaMultiWorker from '@wllama/wllama/src/multi-thread/wllama.worker.mjs?url';
import { InferenceParams, Model } from './utils/types';

export const WLLAMA_CONFIG_PATHS = {
  'single-thread/wllama.js': wllamaSingleJS,
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.js': wllamaMultiJS,
  'multi-thread/wllama.wasm': wllamaMulti,
  'multi-thread/wllama.worker.mjs': wllamaMultiWorker,
};

export const MAX_GGUF_SIZE = 2 * 1024 * 1024 * 1024; // 2GB

export const LIST_MODELS: Model[] = [
  {
    url: 'https://huggingface.co/ngxson/wllama-split-models/resolve/main/gemma-2-2b-it-abliterated-Q4_K_M-00001-of-00004.gguf',
    size: 1708583264,
  },
  {
    url: 'https://huggingface.co/ngxson/wllama-split-models/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M-00001-of-00003.gguf',
    size: 668788416,
  },
  {
    url: 'https://huggingface.co/ngxson/wllama-split-models/resolve/main/qwen2-1_5b-instruct-q4_k_m-00001-of-00004.gguf',
    size: 986046272,
  },
  {
    url: 'https://huggingface.co/ngxson/wllama-split-models/resolve/main/meta-llama-3.1-8b-instruct-abliterated.Q2_K-00001-of-00014.gguf',
    size: 3179133600,
  },
];

export const DEFAULT_INFERENCE_PARAMS: InferenceParams = {
  nThreads: -1, // auto
  nContext: 4096,
  nPredict: 4096,
  nBatch: 128,
  temperature: 0.2,
};

export const DEFAULT_CHAT_TEMPLATE =
  "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
