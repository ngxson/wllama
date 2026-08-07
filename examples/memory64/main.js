import { Wllama } from '../../esm/index.js';

const GIB_BYTES = 1024 ** 3;
const MEMORY64_MAX_PAGES = 262_144n;
const MODEL_PRESETS = {
  tiny: {
    label: 'Harness check · TinyLlama 15M · 18.2 MiB',
    shortLabel: 'Tiny harness check',
    expectedBytes: 19_077_344,
    url: 'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf',
  },
  '4g': {
    label: '4 GiB tier · Qwen2.5 7B Q4_K_M · 4.36 GiB',
    shortLabel: 'Qwen2.5 7B Q4_K_M',
    expectedBytes: 4_683_073_632,
    url: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/bb5d59e06d9551d752d08b292a50eb208b07ab1f/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf',
  },
  '8g': {
    label: '8 GiB tier · Qwen2.5 Coder 14B Q4_K_M · 8.37 GiB',
    shortLabel: 'Qwen2.5 Coder 14B Q4_K_M',
    expectedBytes: 8_988_110_400,
    url: 'https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/d0a692ef765eefbf2fabb130b3cb2e8917e3d225/qwen2.5-coder-14b-instruct-q4_k_m-00001-of-00002.gguf',
  },
  '12g': {
    label: '>8 GiB tier · Qwen2.5 Coder 14B Q6_K · 11.29 GiB',
    shortLabel: 'Qwen2.5 Coder 14B Q6_K',
    expectedBytes: 12_124_683_840,
    url: 'https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/d0a692ef765eefbf2fabb130b3cb2e8917e3d225/qwen2.5-coder-14b-instruct-q6_k-00001-of-00002.gguf',
  },
};

const elements = {
  capability: document.querySelector('#capability'),
  clear: document.querySelector('#clear'),
  clearConsole: document.querySelector('#clear-console'),
  completion: document.querySelector('#completion'),
  console: document.querySelector('#console'),
  downloaded: document.querySelector('#downloaded'),
  elapsed: document.querySelector('#elapsed'),
  message: document.querySelector('#message'),
  modelPreset: document.querySelector('#model-preset'),
  modelSize: document.querySelector('#model-size'),
  modelUrl: document.querySelector('#model-url'),
  outcome: document.querySelector('#outcome'),
  progress: document.querySelector('#progress'),
  run: document.querySelector('#run'),
  stage: document.querySelector('#stage'),
};

const query = new URLSearchParams(location.search);
const initialPreset = MODEL_PRESETS[query.get('case')]
  ? query.get('case')
  : '4g';
const configuredThreads = Number(query.get('threads') || 1);
const configuredTokens = Number(query.get('tokens') || 2);
const configuredStorageQuota = Number(query.get('quota') || 0);
const downloadOnly = query.get('downloadOnly') === '1';
const cacheOnly = query.get('cacheOnly') === '1';
let activeWllama;
let elapsedTimer;

const state = {
  completion: '',
  cacheOnly,
  downloadedBytes: 0,
  elapsedMs: 0,
  error: null,
  expectedBytes: MODEL_PRESETS[initialPreset].expectedBytes,
  finishedAt: null,
  model: MODEL_PRESETS[initialPreset].shortLabel,
  modelUrl: MODEL_PRESETS[initialPreset].url,
  mode: downloadOnly ? 'download-only' : 'load-and-inference',
  metadata: null,
  multithread: null,
  stage: 'idle',
  startedAt: null,
  status: 'idle',
  storage: null,
  threads: configuredThreads,
  totalBytes: 0,
};

const formatBytes = (bytes) => {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';

  const units = ['B', 'KiB', 'MiB', 'GiB'];
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  );
  return `${(bytes / 1024 ** index).toFixed(index < 3 ? 1 : 2)} ${units[index]}`;
};

const formatElapsed = (milliseconds) => {
  const totalSeconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
};

const serializeLogValue = (value) => {
  if (value instanceof Error) return value.stack || value.message;
  if (typeof value === 'string') return value;

  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const appendLog = (level, ...values) => {
  const item = document.createElement('li');
  const timestamp = new Date().toISOString().slice(11, 23);

  item.className = level;
  item.textContent = `[${timestamp}] ${level.toUpperCase()} ${values
    .map(serializeLogValue)
    .join(' ')}`;
  elements.console.append(item);
  item.scrollIntoView({ block: 'nearest' });
};

const logger = {
  debug: (...values) => {
    console.debug(...values);
    appendLog('debug', ...values);
  },
  error: (...values) => {
    console.error(...values);
    appendLog('error', ...values);
  },
  log: (...values) => {
    console.log(...values);
    appendLog('log', ...values);
  },
  warn: (...values) => {
    console.warn(...values);
    appendLog('warn', ...values);
  },
};

const renderState = () => {
  const progress = state.totalBytes
    ? Math.min(100, (state.downloadedBytes / state.totalBytes) * 100)
    : 0;

  elements.stage.textContent = state.stage;
  elements.modelSize.textContent = formatBytes(state.expectedBytes);
  elements.downloaded.textContent = state.totalBytes
    ? `${formatBytes(state.downloadedBytes)} / ${formatBytes(state.totalBytes)}`
    : '—';
  elements.elapsed.textContent = state.startedAt
    ? formatElapsed(state.elapsedMs)
    : '—';
  elements.progress.value = state.stage === 'passed' ? 100 : progress;
  elements.progress.textContent = `${Math.round(progress)}%`;
  elements.completion.textContent = state.completion;
  elements.outcome.textContent =
    state.status === 'running'
      ? 'Running'
      : state.status === 'passed'
        ? 'Passed'
        : state.status === 'failed'
          ? 'Failed'
          : 'Not started';
  elements.outcome.className = `badge ${state.status}`;
};

const setMessage = (message) => {
  elements.message.textContent = message;
  logger.log(message);
};

const assertMemory64 = () => {
  if (!crossOriginIsolated) {
    throw new Error(
      'The page is not cross-origin isolated (COOP/COEP missing)'
    );
  }
  if (!WebAssembly.Suspending || !WebAssembly.promising) {
    throw new Error('JSPI is unavailable; Chromium 137 or newer is required');
  }

  const memory = new WebAssembly.Memory({
    address: 'i64',
    initial: 1n,
    maximum: MEMORY64_MAX_PAGES,
    shared: true,
  });
  if (memory.grow(0n) !== 1n) {
    throw new Error('The shared Memory64 descriptor did not grow correctly');
  }
};

const selectPreset = (key) => {
  const preset = MODEL_PRESETS[key];

  state.expectedBytes = preset.expectedBytes;
  state.model = preset.shortLabel;
  state.modelUrl = preset.url;
  elements.modelUrl.value = preset.url;
  renderState();
};

const clearCache = async () => {
  const wllama = new Wllama(
    { default: '../../esm/wasm/wllama.wasm' },
    { logger }
  );
  wllama.setCompat(null);

  elements.clear.disabled = true;
  try {
    await wllama.cacheManager.clear();
    setMessage('The origin-private model cache is empty.');
  } finally {
    elements.clear.disabled = false;
  }
};

const checkStorageCapacity = async () => {
  const persisted = navigator.storage.persist
    ? await navigator.storage.persist()
    : false;
  const { quota = 0, usage = 0 } = await navigator.storage.estimate();
  const effectiveQuota = Math.max(quota, configuredStorageQuota);
  const availableBytes = Math.max(0, effectiveQuota - usage);

  state.storage = {
    availableBytes,
    browserQuotaBytes: quota,
    persisted,
    quotaBytes: effectiveQuota,
    quotaOverrideBytes: configuredStorageQuota,
    usageBytes: usage,
  };
  logger.log('Browser storage capacity', state.storage);

  const selectedPreset = MODEL_PRESETS[elements.modelPreset.value];
  if (
    state.modelUrl === selectedPreset.url &&
    availableBytes < selectedPreset.expectedBytes
  ) {
    throw new Error(
      `Browser storage has ${availableBytes} bytes available; ${selectedPreset.expectedBytes} bytes are required for this fixture`
    );
  }
};

const finishRun = (status, error = null) => {
  window.clearInterval(elapsedTimer);
  state.elapsedMs = performance.now() - state.startedAt;
  state.error = error
    ? { message: error.message || String(error), stack: error.stack || '' }
    : null;
  state.finishedAt = new Date().toISOString();
  state.stage = status;
  state.status = status;
  elements.run.disabled = false;
  renderState();
};

const runStressTest = async () => {
  if (state.status === 'running') return;

  state.completion = '';
  state.downloadedBytes = 0;
  state.elapsedMs = 0;
  state.error = null;
  state.finishedAt = null;
  state.metadata = null;
  state.multithread = null;
  state.modelUrl = elements.modelUrl.value.trim();
  state.stage = 'capability check';
  state.startedAt = performance.now();
  state.status = 'running';
  state.totalBytes = 0;
  elements.run.disabled = true;
  elapsedTimer = window.setInterval(() => {
    state.elapsedMs = performance.now() - state.startedAt;
    renderState();
  }, 1000);
  renderState();

  try {
    assertMemory64();
    logger.log('Memory64 capability check passed', {
      crossOriginIsolated,
      expectedBytes: state.expectedBytes,
      model: state.model,
      threads: configuredThreads,
    });

    activeWllama = new Wllama(
      { default: '../../esm/wasm/wllama.wasm' },
      { logger, parallelDownloads: 2 }
    );
    activeWllama.setCompat(null);

    if (query.get('clear') === '1') {
      state.stage = 'clearing cache';
      renderState();
      await activeWllama.cacheManager.clear();
    }

    state.stage = 'checking storage';
    renderState();
    await checkStorageCapacity();

    state.stage = 'downloading model';
    setMessage(`Downloading ${state.model}…`);
    const model = cacheOnly
      ? (await activeWllama.modelManager.getModels()).find(
          (candidate) => candidate.url === state.modelUrl
        )
      : await activeWllama.modelManager.getModelOrDownload(
          { url: state.modelUrl },
          {
            progressCallback: ({ loaded, total }) => {
              state.downloadedBytes = loaded;
              state.totalBytes = total;
              renderState();
            },
          }
        );
    if (!model) {
      throw new Error(
        `Cache-only inference could not find a valid cached model for ${state.modelUrl}`
      );
    }
    state.downloadedBytes = model.size;
    state.totalBytes = model.size;
    renderState();

    const selectedPreset = MODEL_PRESETS[elements.modelPreset.value];
    if (
      state.modelUrl === selectedPreset.url &&
      state.totalBytes !== selectedPreset.expectedBytes
    ) {
      throw new Error(
        `Downloaded ${state.totalBytes} bytes; expected exactly ${selectedPreset.expectedBytes} bytes for the pinned fixture`
      );
    }
    if (
      state.expectedBytes > 4 * GIB_BYTES &&
      state.totalBytes <= 4 * GIB_BYTES
    ) {
      throw new Error('The downloaded model did not cross the 4 GiB boundary');
    }

    if (downloadOnly) {
      logger.log('Memory64 fixture download passed', {
        totalBytes: state.totalBytes,
      });
      setMessage('Real model download passed; the cache is ready for loading.');
      finishRun('passed');
      return;
    }

    state.stage = 'loading model';
    setMessage('Download complete. Loading tensors into Memory64…');
    await activeWllama.loadModel(model, {
      n_batch: 32,
      n_ctx: 128,
      n_gpu_layers: 0,
      n_threads: configuredThreads,
      warmup: false,
    });

    const metadata = activeWllama.getModelMetadata();
    if (metadata.hparams.nLayer <= 0 || metadata.hparams.nVocab <= 0) {
      throw new Error('Model load returned invalid metadata');
    }
    state.metadata = metadata.hparams;
    state.multithread = activeWllama.isMultithread();

    state.stage = 'generating tokens';
    setMessage('Model loaded. Generating deterministic tokens…');
    const completion = await activeWllama.createCompletion({
      max_tokens: configuredTokens,
      prompt: 'The capital of France is',
      seed: 42,
      temperature: 0,
    });
    state.completion = completion.choices[0].text;
    if (!state.completion) {
      throw new Error('Inference completed without generating text');
    }

    logger.log('Memory64 stress test passed', {
      completion: state.completion,
      metadata: metadata.hparams,
      totalBytes: state.totalBytes,
    });
    setMessage('Real model loading and inference passed.');
    finishRun('passed');
  } catch (error) {
    logger.error('Memory64 stress test failed', error);
    elements.message.textContent = error.message || String(error);
    finishRun('failed', error);
  } finally {
    await activeWllama?.exit().catch((error) => {
      logger.error('Failed to terminate Wllama cleanly', error);
    });
    activeWllama = undefined;
  }
};

Object.entries(MODEL_PRESETS).forEach(([key, preset]) => {
  const option = document.createElement('option');
  option.value = key;
  option.textContent = preset.label;
  elements.modelPreset.append(option);
});
elements.modelPreset.value = initialPreset;
selectPreset(initialPreset);

elements.modelPreset.addEventListener('change', () => {
  selectPreset(elements.modelPreset.value);
});
elements.modelUrl.addEventListener('input', () => {
  state.modelUrl = elements.modelUrl.value.trim();
});
elements.run.addEventListener('click', runStressTest);
elements.clear.addEventListener('click', clearCache);
elements.clearConsole.addEventListener('click', () => {
  elements.console.replaceChildren();
});

window.__wllamaMemory64Stress = {
  getState: () => structuredClone(state),
  models: structuredClone(MODEL_PRESETS),
  run: runStressTest,
};

try {
  assertMemory64();
  elements.capability.textContent = 'Memory64 + JSPI ready';
  elements.capability.className = 'badge passed';
} catch (error) {
  elements.capability.textContent = 'Capability check failed';
  elements.capability.className = 'badge failed';
  elements.run.disabled = true;
  logger.error(error);
}

if (query.get('autostart') === '1') runStressTest();
