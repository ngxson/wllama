// See: https://vitejs.dev/guide/assets#explicit-url-imports
import wllamaSingleJS from '@wllama/wllama/src/single-thread/wllama.js?url';
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMultiJS from '@wllama/wllama/src/multi-thread/wllama.js?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';
import wllamaMultiWorker from '@wllama/wllama/src/multi-thread/wllama.worker.mjs?url';
import { InferenceParams, Model } from './types';

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
    url: 'https://huggingface.co/ngxson/test_gguf_models/resolve/main/neuralreyna-mini-1.8b-v0.3.q4_k_m-00001-of-00005.gguf',
    size: 100000,
  },
  {
    url: 'https://huggingface.co/ngxson/test_gguf_models/resolve/main/stories15M_moe.gguf',
    size: 100000,
  }
];

export const DEFAULT_INFERENCE_PARAMS: InferenceParams = {
  nThreads: -1, // auto
  nContext: 4096,
  nPredict: 4096,
  nBatch: 128,
  temperature: 0.2,
};

export const DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
