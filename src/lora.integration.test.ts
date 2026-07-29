import { expect, test } from 'vitest';
import { Wllama } from './wllama';

declare const __LORA_INTEGRATION__: boolean;

const CONFIG_PATHS = { default: '/src/wasm/wllama.wasm' };
const BASE_URL =
  'https://huggingface.co/TheBloke/TinyLlama-1.1B-intermediate-step-1431k-3T-GGUF/resolve/0442cb0964eb5dc7a95e9e0ec7a633b4d84085fe/tinyllama-1.1b-intermediate-step-1431k-3t.Q4_K_M.gguf';
const ADAPTER_URL =
  'https://huggingface.co/Dips-1991/tinyllama-1.1B_alpaca_2k_lora-F16-GGUF/resolve/2b45be40c0e190a62130a04b2386a67bd5c3291b/tinyllama-1.1B_alpaca_2k_lora-f16.gguf';

const integrationTest = __LORA_INTEGRATION__ ? test : test.skip;

integrationTest(
  'switches base -> LoRA -> base without reloading the model',
  async () => {
    const adapterResponse = await fetch(ADAPTER_URL);
    expect(adapterResponse.ok).toBe(true);
    const adapter = await adapterResponse.blob();

    const wllama = new Wllama(CONFIG_PATHS, { suppressNativeLog: false });
    wllama.setCompat(null);
    await wllama.loadModelFromUrl(BASE_URL, {
      n_ctx: 512,
      n_batch: 256,
      n_gpu_layers: 99999,
      lora_adapters: [{ blob: adapter, scale: 1 }],
      lora_init_without_apply: true,
    });

    const request = {
      prompt:
        'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat are the three primary colors?\n\n### Response:\n',
      max_tokens: 64,
      temperature: 0,
      seed: 42,
      cache_prompt: false,
      top_k: 1,
      stop: ['### Instruction:', '[end of text]'],
    };

    const baseBefore = await wllama.createCompletion({ ...request, lora: [{ id: 0, scale: 0 }] });
    const adapted = await wllama.createCompletion({
      ...request,
      lora: [{ id: 0, scale: 1 }],
    });
    const baseAfter = await wllama.createCompletion({ ...request, lora: [{ id: 0, scale: 0 }] });

    const beforeText = baseBefore.choices[0]?.text;
    const adaptedText = adapted.choices[0]?.text;
    const afterText = baseAfter.choices[0]?.text;
    console.log({ beforeText, adaptedText, afterText });
    expect(beforeText).toBeTruthy();
    expect(adaptedText).toBeTruthy();
    expect(adaptedText).not.toBe(beforeText);
    expect(afterText).toBe(beforeText);

    await wllama.exit();
  },
  300_000
);