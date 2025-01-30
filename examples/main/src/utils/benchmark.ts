import { Wllama } from '@wllama/wllama';
import { WLLAMA_CONFIG_PATHS } from '../config';
import { delay } from './utils';

// TODO: this is console-only for now, should we implement a GUI in the future?

const WIKITEXT_URL =
  'https://raw.githubusercontent.com/wangfin/QAsystem/refs/heads/master/QAManagement/language_model/data/wikitext-2/valid.txt';

const BENCH_MODELS = [
  'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf',
  'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf',
  'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
  'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_L.gguf',
];

const BENCH_N_REPEATED = 4;

const BENCH_CONFIGS: { type: 'pp' | 'tg'; n_samples: number }[] = [
  { type: 'pp', n_samples: 32 },
  { type: 'pp', n_samples: 64 },
  { type: 'pp', n_samples: 128 },
  { type: 'pp', n_samples: 256 },
  { type: 'tg', n_samples: 32 },
  { type: 'tg', n_samples: 64 },
  { type: 'tg', n_samples: 128 },
  { type: 'tg', n_samples: 256 },
];

async function loadModel(modelUrl: string) {
  const modelFile = modelUrl.split('/').pop();
  const wllama = new Wllama(WLLAMA_CONFIG_PATHS);
  await wllama.loadModelFromUrl(modelUrl, {
    n_batch: 512,
    n_ctx: 4096,
    progressCallback: ({ total, loaded }) => {
      console.log(`Model ${modelFile}: ${Math.round((100 * loaded) / total)}%`);
    },
  });
  return { wllama, modelFile };
}

async function benchmark() {
  const output: any[][] = [
    ['model', 'threads', 'test', 't/s'],
    ['---', '---', '---', '---'],
  ];
  for (const modelUrl of BENCH_MODELS) {
    const [{ wllama, modelFile }] = await Promise.all([
      loadModel(modelUrl),
      delay(10000), // force delay for CPU to cool down
    ]);
    console.clear();
    const nThreads = wllama.getNumThreads();
    for (const config of BENCH_CONFIGS) {
      const { type, n_samples } = config;
      const results: number[] = [];
      for (let i = 0; i < BENCH_N_REPEATED; i++) {
        console.log('Running', modelFile, config);
        const { t_ms } = await wllama._testBenchmark(type, n_samples);
        const t_per_tok = n_samples / (t_ms / 1000);
        results.push(t_per_tok);
        console.log('Run ', i, 'pref:', t_per_tok, 't/s');
      }
      const t_avg = results.reduce((a, b) => a + b, 0) / results.length;
      const t_plus_minus = Math.abs(
        Math.max(...results) - Math.min(...results)
      );
      output.push([
        modelFile,
        nThreads,
        `${type} ${n_samples}`,
        `${t_avg.toFixed(2)} ± ${t_plus_minus.toFixed(2)}`,
      ]);
    }
    wllama.exit();
  }

  console.table(output);
  const markdown = output
    .map((row) => '| ' + row.join(' | ') + ' |')
    .join('\n');
  console.log(markdown);
}

async function perplexity() {
  const output: any[][] = [
    ['model', 'PPL', 'n_tokens'],
    ['---', '---', '---'],
  ];
  const LIMIT_TOKENS = 2048;
  const wikitext = await fetch(WIKITEXT_URL).then((res) => res.text());
  console.log('Loaded wikitext:', wikitext.substring(0, 100), '...');
  for (const modelUrl of BENCH_MODELS) {
    const { wllama, modelFile } = await loadModel(modelUrl);
    console.clear();
    let tokens = await wllama.tokenize(
      wikitext.substring(0, LIMIT_TOKENS * 16)
    );
    tokens = tokens.slice(0, LIMIT_TOKENS);
    console.log('Running', modelFile, 'n_tokens', tokens.length);
    const { ppl } = await wllama._testPerplexity(tokens);
    console.log('PPL:', ppl);
    output.push([modelFile, ppl, tokens.length]);
    wllama.exit();
  }

  console.table(output);
  const markdown = output
    .map((row) => '| ' + row.join(' | ') + ' |')
    .join('\n');
  console.log(markdown);
}

(window as any).__benchmark = benchmark;
(window as any).__perplexity = perplexity;
