// See: https://vitejs.dev/guide/assets#explicit-url-imports
import wllamaSingleJS from '@wllama/wllama/src/single-thread/wllama.js?url';
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMultiJS from '@wllama/wllama/src/multi-thread/wllama.js?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';
import wllamaMultiWorker from '@wllama/wllama/src/multi-thread/wllama.worker.mjs?url';
import { InferenceParams, Message, Model } from './types';

export const WLLAMA_CONFIG_PATHS = {
  'single-thread/wllama.js': wllamaSingleJS,
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.js': wllamaMultiJS,
  'multi-thread/wllama.wasm': wllamaMulti,
  'multi-thread/wllama.worker.mjs': wllamaMultiWorker,
};

function formatChatML(messages: Message[]): string {
  let output = '';
  for (const msg of messages) {
    output += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
  }
  output += '<|im_start|>assistant\n';
  return output;
}

export const LIST_MODELS: Model[] = [
  {
    url: 'https://huggingface.co/MaziyarPanahi/SmolLM-1.7B-Instruct-GGUF/resolve/main/SmolLM-1.7B-Instruct.Q4_K_M.gguf',
    size: '1.06 GB',
    formatChat: formatChatML,
  },
  {
    url: 'https://huggingface.co/ngxson/test_gguf_models/resolve/main/stories15M_moe.gguf',
    size: '73.5 MB',
    formatChat: formatChatML,
  }
];

export const DEFAULT_INFERENCE_PARAMS: InferenceParams = {
  nThreads: -1, // auto
  nContext: 4096,
  nPredict: 10, // TODO: change me
  temperature: 0.2,
};
