// See: https://vitejs.dev/guide/assets#explicit-url-imports
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';
import wllamaPackageJson from '@wllama/wllama/package.json';
import { InferenceParams } from './utils/types';

export const WLLAMA_VERSION = wllamaPackageJson.version;

export const WLLAMA_CONFIG_PATHS = {
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.wasm': wllamaMulti,
};

export const MAX_GGUF_SIZE = 2 * 1024 * 1024 * 1024; // 2GB

export const LIST_MODELS = [
  {
    url: 'https://huggingface.co/ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf',
    size: 386404992,
  },
  {
    url: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf',
    size: 675710816,
  },
  {
    url: 'https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-1b-instruct-q4_k_m.gguf',
    size: 807690656,
  },
  {
    url: 'https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf',
    size: 924456032,
  },
  {
    url: 'https://huggingface.co/ngxson/wllama-split-models/resolve/main/qwen2-1_5b-instruct-q4_k_m-00001-of-00004.gguf',
    size: 986046272,
  },
  {
    url: 'https://huggingface.co/ngxson/SmolLM2-1.7B-Instruct-Q4_K_M-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf',
    size: 1055609536,
  },
  {
    url: 'https://huggingface.co/ngxson/wllama-split-models/resolve/main/gemma-2-2b-it-abliterated-Q4_K_M-00001-of-00004.gguf',
    size: 1708583264,
  },
  {
    url: 'https://huggingface.co/ngxson/wllama-split-models/resolve/main/neuralreyna-mini-1.8b-v0.3.q4_k_m-00001-of-00005.gguf',
    size: 1217753472,
  },
  {
    url: 'https://huggingface.co/ngxson/wllama-split-models/resolve/main/Phi-3.1-mini-128k-instruct-Q3_K_M-00001-of-00008.gguf',
    size: 1955478176,
  },
  {
    url: 'https://huggingface.co/ngxson/wllama-split-models/resolve/main/meta-llama-3.1-8b-instruct-abliterated.Q2_K-00001-of-00014.gguf',
    size: 3179133600,
  },
  {
    url: 'https://huggingface.co/ngxson/wllama-split-models/resolve/main/Meta-Llama-3.1-8B-Instruct-Q2_K-00001-of-00014.gguf',
    size: 3179138048,
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
