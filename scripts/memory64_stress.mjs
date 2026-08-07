import { execFileSync } from 'node:child_process';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import express from 'express';
import { chromium } from 'playwright';

import {
  closeAndCheckBrowserEvents,
  createFailureLatch,
  createSingleFlight,
  matchesExpectedCompletion,
  resolveCgroupMount,
} from './memory64_stress_helpers.mjs';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDir, '..');
const GIB_BYTES = 1024 ** 3;
const FOUR_GIB_BYTES = 4 * GIB_BYTES;
const ASSUMED_AVAILABLE_BYTES = 12 * GIB_BYTES;
const ASYNC_FILE_READ_CHUNK_BYTES = 64 * 1024 ** 2;
const DEFAULT_TIMEOUT_MS = 3 * 60 * 60 * 1000;
const STRESS_STORAGE_QUOTA_BYTES = 32 * GIB_BYTES;
const MODEL_CASES = {
  tiny: {
    expectedCompletion: ' a very',
    expectedBytes: 19_077_344,
    label: 'Tiny harness check',
  },
  '4g': {
    expectedCompletionSubstring: 'Paris',
    expectedBytes: 5_011_843_904,
    label: 'Gemma 7B Q4_0',
  },
  '8g': {
    expectedCompletion: ' Paris.',
    expectedBytes: 8_988_110_400,
    label: 'Qwen2.5 Coder 14B Q4_K_M',
  },
  '12g': {
    expectedCompletion: ' Paris.',
    expectedBytes: 12_124_683_840,
    label: 'Qwen2.5 Coder 14B Q6_K',
  },
};
const FATAL_CONSOLE_PATTERN = new RegExp(
  [
    'Aborted',
    'Cannot allocate WebAssembly\\.Memory',
    'Cannot enlarge memory',
    'CompileError',
    'DataCloneError',
    'failed to allocate',
    'failed to grow',
    'failed to load model',
    'File read failed',
    'LinkError',
    'memory access out of bounds',
    'out of memory',
    'RangeError',
    'RuntimeError',
    'std::bad_alloc',
    'worker sent an error',
  ].join('|'),
  'i'
);

const timestamp = new Date().toISOString().replaceAll(/[:.]/g, '-');
const artifactRoot =
  process.env.WLLAMA_STRESS_ARTIFACTS ||
  join(tmpdir(), 'wllama-memory64-artifacts', timestamp);
const selectedCases = (
  process.env.WLLAMA_STRESS_CASES ||
  process.argv.find((argument) => argument.startsWith('--cases='))?.slice(8) ||
  '4g,8g,12g'
)
  .split(',')
  .map((value) => value.trim())
  .filter(Boolean);
const defaultRunOptions = {
  downloadOnly: process.env.WLLAMA_STRESS_DOWNLOAD_ONLY === '1',
  keepBrowserProfile: process.env.WLLAMA_STRESS_KEEP_PROFILE === '1',
  reuseBrowserProfile: process.env.WLLAMA_STRESS_REUSE_PROFILE === '1',
};
const twoPhase = process.env.WLLAMA_STRESS_TWO_PHASE === '1';
const serverPort = Number(process.env.WLLAMA_STRESS_PORT) || 0;
const stressThreads = Number(process.env.WLLAMA_STRESS_THREADS) || 1;
const timeoutMs =
  Number(process.env.WLLAMA_STRESS_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS;

if (!Number.isSafeInteger(stressThreads) || stressThreads < 1) {
  throw new Error('WLLAMA_STRESS_THREADS must be a positive integer');
}

const writeLine = (message) => process.stdout.write(`${message}\n`);
const formatGiB = (bytes) => `${(bytes / GIB_BYTES).toFixed(2)} GiB`;
const now = () => new Date().toISOString();

const readMeminfo = async () => {
  const content = await readFile('/proc/meminfo', 'utf8');
  const values = Object.fromEntries(
    content.split('\n').flatMap((line) => {
      const match = line.match(/^(\w+):\s+(\d+) kB$/);
      return match ? [[match[1], Number(match[2]) * 1024]] : [];
    })
  );

  return {
    availableBytes: values.MemAvailable,
    freeBytes: values.MemFree,
    totalBytes: values.MemTotal,
  };
};

const unknownCgroupMemory = () => ({
  availableBytes: null,
  currentBytes: null,
  maximumBytes: null,
  path: null,
  resolved: null,
  unlimited: null,
  version: null,
});

const unresolvedCgroupMemory = (version) => ({
  availableBytes: null,
  currentBytes: null,
  maximumBytes: null,
  path: null,
  resolved: false,
  unlimited: false,
  version,
});

const readCgroupUsage = async (path, currentFile, maximumFile, version) => {
  const [current, maximum] = await Promise.all([
    readFile(join(path, currentFile), 'utf8'),
    readFile(join(path, maximumFile), 'utf8'),
  ]);
  const currentBytes = Number(current.trim());
  const maximumValue = maximum.trim();
  const maximumBigInt = maximumValue === 'max' ? null : BigInt(maximumValue);
  const unlimited = maximumBigInt === null || maximumBigInt >= 1n << 60n;

  if (unlimited) {
    return {
      availableBytes: null,
      currentBytes,
      maximumBytes: null,
      path,
      resolved: true,
      unlimited: true,
      version,
    };
  }

  const maximumBytes = Number(maximumBigInt);
  return {
    availableBytes: Math.max(0, maximumBytes - currentBytes),
    currentBytes,
    maximumBytes,
    path,
    resolved: true,
    unlimited: false,
    version,
  };
};

const readCgroupMemory = async () => {
  try {
    const [membership, mountInfo] = await Promise.all([
      readFile('/proc/self/cgroup', 'utf8'),
      readFile('/proc/self/mountinfo', 'utf8'),
    ]);

    const v2Entry = membership
      .split('\n')
      .find((line) => line.startsWith('0::'));
    if (v2Entry) {
      const { path } = resolveCgroupMount(
        mountInfo,
        'cgroup2',
        v2Entry.slice(3) || '/'
      );
      if (path) {
        return await readCgroupUsage(
          path,
          'memory.current',
          'memory.max',
          2
        ).catch(() => unresolvedCgroupMemory(2));
      }
      return unresolvedCgroupMemory(2);
    }

    const v1Entry = membership.split('\n').find((line) => {
      const controllers = line.split(':')[1]?.split(',') || [];
      return controllers.includes('memory');
    });
    if (!v1Entry) return unknownCgroupMemory();

    const { path } = resolveCgroupMount(
      mountInfo,
      'cgroup',
      v1Entry.split(':').slice(2).join(':') || '/',
      'memory'
    );
    return path
      ? await readCgroupUsage(
          path,
          'memory.usage_in_bytes',
          'memory.limit_in_bytes',
          1
        ).catch(() => unresolvedCgroupMemory(1))
      : unresolvedCgroupMemory(1);
  } catch {
    return unresolvedCgroupMemory(null);
  }
};

const readProcessTree = async (rootPid) => {
  const seen = new Set();

  const visit = async (pid) => {
    if (seen.has(pid)) return;
    seen.add(pid);

    try {
      const children = await readFile(
        `/proc/${pid}/task/${pid}/children`,
        'utf8'
      );
      await Promise.all(
        children
          .trim()
          .split(/\s+/)
          .filter(Boolean)
          .map((child) => visit(Number(child)))
      );
    } catch {
      // A short-lived browser subprocess may disappear between samples.
    }
  };

  await visit(rootPid);
  return [...seen];
};

const readBrowserMemory = async (rootPid) => {
  const pids = await readProcessTree(rootPid);
  const totals = { pssBytes: 0, rssBytes: 0, swapBytes: 0 };

  await Promise.all(
    pids.map(async (pid) => {
      try {
        const rollup = await readFile(`/proc/${pid}/smaps_rollup`, 'utf8');
        const readKilobytes = (name) =>
          Number(
            rollup.match(new RegExp(`^${name}:\\s+(\\d+) kB$`, 'm'))?.[1] || 0
          );

        totals.pssBytes += readKilobytes('Pss') * 1024;
        totals.rssBytes += readKilobytes('Rss') * 1024;
        totals.swapBytes += readKilobytes('Swap') * 1024;
      } catch {
        // The sample remains useful when one short-lived process exits.
      }
    })
  );

  return { ...totals, processCount: pids.length };
};

const createServer = async () => {
  const app = express();

  app.use(
    express.static(projectRoot, {
      setHeaders(response) {
        response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
        response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
        response.setHeader(
          'Cache-Control',
          'no-cache, no-store, must-revalidate'
        );
      },
    })
  );

  return await new Promise((resolvePromise, rejectPromise) => {
    const server = app.listen(serverPort, '127.0.0.1', () => {
      const address = server.address();
      resolvePromise({
        close: () =>
          new Promise((resolveClose, rejectClose) => {
            server.close((error) =>
              error ? rejectClose(error) : resolveClose()
            );
          }),
        origin: `http://127.0.0.1:${address.port}`,
      });
    });
    server.on('error', rejectPromise);
  });
};

const remoteValue = (argument) => {
  if ('value' in argument) return argument.value;
  return argument.description || argument.unserializableValue || argument.type;
};

const getBrowserProfile = (caseName) =>
  join(projectRoot, '.memory64-stress-profiles', caseName);

const activeContexts = new Set();
const activeRunFailures = new Set();
let receivedSignal = null;

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, () => {
    receivedSignal ||= signal;
    for (const failRun of activeRunFailures) {
      failRun(`Stress run interrupted by ${signal}`);
    }
    for (const activeContext of activeContexts) {
      activeContext.close().catch(() => {});
    }
  });
}

const runCase = async (caseName, modelCase, server, runOptions = {}) => {
  const {
    artifactName = caseName,
    downloadOnly,
    keepBrowserProfile,
    reuseBrowserProfile,
  } = { ...defaultRunOptions, ...runOptions };
  const caseArtifacts = join(artifactRoot, artifactName);
  const browserProfile = getBrowserProfile(caseName);
  const cacheOnly = reuseBrowserProfile && !downloadOnly;
  const events = [];
  const memorySamples = [];
  let browser;
  let context;
  let closing = false;
  let meminfo;
  let cgroupMemory;
  let sampleTimer;
  let waitForMemorySample = async () => {};
  const runFailure = createFailureLatch();
  const failBrowser = runFailure.fail;
  const guardBrowser = runFailure.guard;

  await mkdir(caseArtifacts, { recursive: true });
  const record = (kind, detail = {}) => {
    const event = { at: now(), kind, ...detail };
    events.push(event);
    if (detail.fatal) {
      writeLine(`[${caseName}] ${kind}: ${detail.text || detail.error || ''}`);
    }
  };

  activeRunFailures.add(failBrowser);
  try {
    if (receivedSignal) {
      failBrowser(`Stress run interrupted by ${receivedSignal}`);
    }
    runFailure.throwIfFailed();

    [meminfo, cgroupMemory] = await guardBrowser(
      Promise.all([
        readMeminfo().catch(() => null),
        readCgroupMemory().catch(() => unknownCgroupMemory()),
      ])
    );
    runFailure.throwIfFailed();
    await guardBrowser(mkdir(dirname(browserProfile), { recursive: true }));
    if (!reuseBrowserProfile) {
      runFailure.throwIfFailed();
      await guardBrowser(rm(browserProfile, { force: true, recursive: true }));
    }
    runFailure.throwIfFailed();
    const launchPromise = chromium.launchPersistentContext(browserProfile, {
      args: [
        '--disable-dev-shm-usage',
        '--disable-setuid-sandbox',
        '--enable-precise-memory-info',
        '--no-sandbox',
      ],
      env: { ...process.env, TMPDIR: join(tmpdir(), 'wllama-chromium') },
      headless: true,
      viewport: { height: 900, width: 1440 },
    });
    launchPromise
      .then((launchedContext) => {
        if (runFailure.getFailure()) launchedContext.close().catch(() => {});
      })
      .catch(() => {});
    context = await guardBrowser(launchPromise);
    activeContexts.add(context);
    browser = context.browser();
    if (!browser) throw new Error('Persistent Chromium context has no browser');
    browser.on('disconnected', () => {
      if (!closing) {
        record('browser.disconnected', { fatal: true });
        failBrowser('Chromium disconnected during the stress run');
      }
    });
    await guardBrowser(
      context.tracing.start({ screenshots: true, snapshots: true })
    );
    await context.addInitScript(() => {
      addEventListener('error', (event) => {
        console.error('WINDOW_ERROR', event.error || event.message);
      });
      addEventListener('unhandledrejection', (event) => {
        console.error('UNHANDLED_REJECTION', event.reason);
      });
    });

    context.on('console', (message) => {
      const text = message.text();
      const fatal =
        message.type() === 'error' || FATAL_CONSOLE_PATTERN.test(text);
      record('console', {
        fatal,
        level: message.type(),
        location: message.location(),
        text,
      });
      if (fatal) failBrowser(`Fatal browser console message: ${text}`);
    });
    context.on('weberror', (webError) => {
      const error = webError.error().stack || webError.error().message;
      record('context.weberror', {
        error,
        fatal: true,
      });
      failBrowser(`Unhandled browser error: ${error}`);
    });
    context.on('request', (request) => {
      const requestUrl = request.url();
      const protocol = new URL(requestUrl).protocol;
      const remoteNetworkRequest =
        ['http:', 'https:'].includes(protocol) &&
        !requestUrl.startsWith(server.origin);
      if (cacheOnly && remoteNetworkRequest) {
        record('network.cacheOnlyViolation', {
          fatal: true,
          text: requestUrl,
        });
        failBrowser(
          `Cache-only inference attempted a network request: ${requestUrl}`
        );
      }
    });
    context.on('requestfailed', (request) => {
      if (!closing) {
        const error = request.failure()?.errorText;
        const fatal = error !== 'net::ERR_ABORTED';
        record('network.requestfailed', {
          error,
          // Chromium reports intentional response-body cancellation as an
          // aborted request. A real download failure also fails the app state.
          fatal,
          text: request.url(),
        });
        if (fatal)
          failBrowser(`Browser request failed: ${error || request.url()}`);
      }
    });
    context.on('response', (response) => {
      if (response.status() >= 400) {
        record('network.response', {
          fatal: true,
          status: response.status(),
          text: response.url(),
        });
        failBrowser(
          `Browser request returned HTTP ${response.status()}: ${response.url()}`
        );
      }
    });

    const page = context.pages()[0] || (await context.newPage());
    page.on('crash', () => {
      record('page.crash', { fatal: true });
      failBrowser('Chromium renderer crashed during the stress run');
    });
    page.on('pageerror', (error) => {
      const detail = error.stack || error.message;
      record('page.error', { error: detail, fatal: true });
      failBrowser(`Unhandled page error: ${detail}`);
    });
    page.on('worker', (worker) => {
      record('worker.attached', { text: worker.url() });
      worker.on('close', () => record('worker.closed', { text: worker.url() }));
      worker.on('console', (message) => {
        const text = message.text();
        const fatal =
          message.type() === 'error' || FATAL_CONSOLE_PATTERN.test(text);
        record('worker.console', {
          fatal,
          level: message.type(),
          text,
        });
        if (fatal) failBrowser(`Fatal worker console message: ${text}`);
      });
    });

    const browserCdp = await browser.newBrowserCDPSession();
    await browserCdp.send('Target.setDiscoverTargets', { discover: true });
    browserCdp.on('Target.targetCrashed', (event) => {
      record('cdp.targetCrashed', { fatal: true, ...event });
      failBrowser(
        `Chromium target crashed with status ${event.status || 'unknown'}`
      );
    });
    const { processInfo } = await browserCdp.send('SystemInfo.getProcessInfo');
    const browserPid = Number(
      processInfo.find((process) => process.type === 'browser')?.id
    );
    if (!Number.isSafeInteger(browserPid) || browserPid <= 0) {
      throw new Error(
        'Could not identify the Chromium browser process through CDP'
      );
    }
    record('browser.process', { pid: browserPid });

    const pageCdp = await context.newCDPSession(page);
    await Promise.all([
      pageCdp.send('Log.enable'),
      pageCdp.send('Network.enable'),
      pageCdp.send('Performance.enable'),
      pageCdp.send('Runtime.enable'),
    ]);
    await pageCdp.send('Inspector.enable').catch(() => {});
    pageCdp.on('Inspector.targetCrashed', () => {
      record('cdp.inspectorTargetCrashed', { fatal: true });
      failBrowser('Chrome DevTools reported a crashed renderer target');
    });
    pageCdp.on('Log.entryAdded', ({ entry }) => {
      const fatal =
        entry.level === 'error' || FATAL_CONSOLE_PATTERN.test(entry.text);
      record('cdp.log', {
        fatal,
        level: entry.level,
        text: entry.text,
      });
      if (fatal) failBrowser(`Fatal DevTools log entry: ${entry.text}`);
    });
    pageCdp.on('Network.loadingFailed', (event) => {
      if (!closing && !event.canceled) {
        record('cdp.networkLoadingFailed', {
          error: event.errorText,
          fatal: true,
        });
        failBrowser(`DevTools reported a failed request: ${event.errorText}`);
      }
    });
    pageCdp.on('Runtime.consoleAPICalled', (event) => {
      const text = event.args.map(remoteValue).join(' ');
      const fatal = event.type === 'error' || FATAL_CONSOLE_PATTERN.test(text);
      record('cdp.console', {
        fatal,
        level: event.type,
        text,
      });
      if (fatal) failBrowser(`Fatal DevTools console message: ${text}`);
    });
    pageCdp.on('Runtime.exceptionThrown', ({ exceptionDetails }) => {
      const error =
        exceptionDetails.exception?.description ||
        exceptionDetails.text ||
        'Unknown runtime exception';
      record('cdp.exception', {
        error,
        fatal: true,
      });
      failBrowser(`DevTools reported a runtime exception: ${error}`);
    });
    await pageCdp.send('Storage.overrideQuotaForOrigin', {
      origin: server.origin,
      quotaSize: STRESS_STORAGE_QUOTA_BYTES,
    });
    record('cdp.storageQuotaOverride', {
      bytes: STRESS_STORAGE_QUOTA_BYTES,
    });

    const memorySampler = createSingleFlight(async () => {
      const [browserMemory, hostMemory, currentCgroupMemory] =
        await Promise.all([
          readBrowserMemory(browserPid),
          readMeminfo().catch(() => null),
          readCgroupMemory().catch(() => unknownCgroupMemory()),
        ]);
      memorySamples.push({
        at: now(),
        browser: browserMemory,
        cgroup: currentCgroupMemory,
        host: hostMemory,
      });
    });
    const sampleMemory = memorySampler.run;
    waitForMemorySample = memorySampler.wait;
    await sampleMemory();
    sampleTimer = setInterval(() => {
      sampleMemory().catch(failBrowser);
    }, 2000);

    const clearCache = reuseBrowserProfile ? 0 : 1;
    const url = `${server.origin}/examples/memory64/?case=${caseName}&clear=${clearCache}&threads=${stressThreads}&tokens=2&quota=${STRESS_STORAGE_QUOTA_BYTES}&downloadOnly=${downloadOnly ? 1 : 0}&cacheOnly=${cacheOnly ? 1 : 0}`;
    writeLine(`[${caseName}] Opening ${url}`);
    await guardBrowser(page.goto(url, { waitUntil: 'networkidle' }));
    const storageQuota = await pageCdp.send('Storage.getUsageAndQuota', {
      origin: server.origin,
    });
    record('cdp.storageQuota', storageQuota);
    if (
      !storageQuota.overrideActive ||
      storageQuota.quota < modelCase.expectedBytes
    ) {
      throw new Error(
        `Chromium storage quota override is not active for ${caseName}`
      );
    }
    await guardBrowser(
      page.waitForFunction(() => !!window.__wllamaMemory64Stress)
    );
    await guardBrowser(
      page.screenshot({
        fullPage: true,
        path: join(caseArtifacts, 'before.png'),
      })
    );

    // Exercise the proof-of-concept through the same controls a user sees.
    await page.selectOption('#model-preset', caseName);
    await page.click('#run');
    await guardBrowser(
      page.waitForFunction(
        () =>
          ['passed', 'failed'].includes(
            window.__wllamaMemory64Stress.getState().status
          ),
        null,
        { timeout: timeoutMs }
      )
    );
    const appState = await page.evaluate(() =>
      window.__wllamaMemory64Stress.getState()
    );
    await page.waitForTimeout(2000);
    await sampleMemory();
    const preciseMemory = await page
      .evaluate(async () => {
        if (!performance.measureUserAgentSpecificMemory) return null;
        return await performance.measureUserAgentSpecificMemory();
      })
      .catch(() => null);
    const performanceMetrics = await pageCdp.send('Performance.getMetrics');
    await guardBrowser(
      page.screenshot({
        fullPage: true,
        path: join(caseArtifacts, 'after.png'),
      })
    );

    // Close every source of console, CDP, and crash events before snapshotting
    // them. A fatal event delivered during shutdown must still fail this run.
    const browserVersion = browser.version();
    const closingContext = context;
    clearInterval(sampleTimer);
    sampleTimer = undefined;
    await guardBrowser(sampleMemory());
    closing = true;
    await closeAndCheckBrowserEvents({
      close: () => closingContext.close(),
      getFailure: runFailure.getFailure,
      onClosed: () => {
        activeContexts.delete(closingContext);
        context = undefined;
        browser = undefined;
      },
      stopTracing: () =>
        closingContext.tracing.stop({
          path: join(caseArtifacts, 'trace.zip'),
        }),
    });

    const fatalEvents = events.filter((event) => event.fatal);
    const peakBrowserPssBytes = Math.max(
      ...memorySamples.map((sample) => sample.browser.pssBytes)
    );
    const peakBrowserRssBytes = Math.max(
      ...memorySamples.map((sample) => sample.browser.rssBytes)
    );
    const downloadedExactFixture =
      appState.totalBytes === modelCase.expectedBytes &&
      appState.downloadedBytes === modelCase.expectedBytes;
    const asyncFileReadSizes = events.flatMap(({ text = '' }) => {
      const match = text.match(/Largest async file read: (\d+) bytes/);
      return match ? [Number(match[1])] : [];
    });
    const boundedAsyncFileReads =
      asyncFileReadSizes.length > 0 &&
      asyncFileReadSizes.every(
        (readSize) => readSize <= ASYNC_FILE_READ_CHUNK_BYTES
      );
    const fullSharedMemory =
      stressThreads === 1 ||
      events.some(({ text }) => text === 'WASM memory maximum: 16384 MiB');
    const minimumPhysicalPssBytes =
      modelCase.expectedBytes > FOUR_GIB_BYTES ? modelCase.expectedBytes : 1;
    const validInference = downloadOnly
      ? appState.mode === 'download-only'
      : appState.mode === 'load-and-inference' &&
        appState.cacheOnly === cacheOnly &&
        appState.metadata?.nLayer > 0 &&
        appState.metadata?.nVocab > 0 &&
        appState.threads === stressThreads &&
        appState.multithread === stressThreads > 1 &&
        boundedAsyncFileReads &&
        fullSharedMemory &&
        matchesExpectedCompletion(appState.completion, modelCase) &&
        Number.isFinite(peakBrowserPssBytes) &&
        peakBrowserPssBytes >= minimumPhysicalPssBytes;
    const result = {
      appState,
      asyncFileReadSizes,
      boundedAsyncFileReads,
      browserVersion,
      cacheOnly,
      case: caseName,
      configuredThreads: stressThreads,
      downloadedExactFixture,
      expectedBytes: modelCase.expectedBytes,
      expectedCompletion: modelCase.expectedCompletion ?? null,
      expectedCompletionSubstring:
        modelCase.expectedCompletionSubstring ?? null,
      fatalEventCount: fatalEvents.length,
      finishedAt: now(),
      fullSharedMemory,
      label: modelCase.label,
      minimumPhysicalPssBytes,
      mode: downloadOnly ? 'download-only' : 'load-and-inference',
      peakBrowserPssBytes,
      peakBrowserRssBytes,
      performanceMetrics,
      preciseMemory,
      assumedAvailableBytes: ASSUMED_AVAILABLE_BYTES,
      startedWithAvailableBytes: meminfo?.availableBytes ?? null,
      startedWithCgroupMemory: cgroupMemory,
      status:
        appState.status === 'passed' &&
        downloadedExactFixture &&
        validInference &&
        fatalEvents.length === 0
          ? 'passed'
          : 'failed',
      validInference,
    };

    await Promise.all([
      writeFile(
        join(caseArtifacts, 'events.json'),
        `${JSON.stringify(events, null, 2)}\n`
      ),
      writeFile(
        join(caseArtifacts, 'memory.json'),
        `${JSON.stringify(memorySamples, null, 2)}\n`
      ),
      writeFile(
        join(caseArtifacts, 'result.json'),
        `${JSON.stringify(result, null, 2)}\n`
      ),
    ]);
    writeLine(
      `[${caseName}] ${result.status.toUpperCase()} peak browser PSS ${formatGiB(result.peakBrowserPssBytes)}`
    );
    return result;
  } catch (error) {
    clearInterval(sampleTimer);
    sampleTimer = undefined;
    await waitForMemorySample().catch(() => {});
    const result = {
      case: caseName,
      configuredThreads: stressThreads,
      error: { message: error.message, stack: error.stack },
      expectedBytes: modelCase.expectedBytes,
      fatalEventCount: events.filter((event) => event.fatal).length,
      mode: downloadOnly ? 'download-only' : 'load-and-inference',
      status: 'failed',
    };
    await Promise.all([
      writeFile(
        join(caseArtifacts, 'events.json'),
        `${JSON.stringify(events, null, 2)}\n`
      ),
      writeFile(
        join(caseArtifacts, 'memory.json'),
        `${JSON.stringify(memorySamples, null, 2)}\n`
      ),
      writeFile(
        join(caseArtifacts, 'result.json'),
        `${JSON.stringify(result, null, 2)}\n`
      ),
    ]);
    writeLine(`[${caseName}] FAILED ${error.stack || error.message}`);
    return result;
  } finally {
    closing = true;
    activeRunFailures.delete(failBrowser);
    clearInterval(sampleTimer);
    await waitForMemorySample().catch(() => {});
    if (context) {
      await context.tracing
        .stop({ path: join(caseArtifacts, 'trace.zip') })
        .catch(() => {});
      await context.close().catch(() => {});
      activeContexts.delete(context);
    }
    await browser?.close().catch(() => {});
    if (!keepBrowserProfile) {
      await rm(browserProfile, { force: true, recursive: true });
    }
  }
};

const invalidCases = selectedCases.filter((caseName) => !MODEL_CASES[caseName]);
if (invalidCases.length > 0) {
  throw new Error(`Unknown stress cases: ${invalidCases.join(', ')}`);
}
const oversizedCases = selectedCases.filter(
  (caseName) => MODEL_CASES[caseName].expectedBytes > ASSUMED_AVAILABLE_BYTES
);
if (oversizedCases.length > 0) {
  throw new Error(
    `Stress cases exceed the ${formatGiB(ASSUMED_AVAILABLE_BYTES)} test budget: ${oversizedCases.join(', ')}`
  );
}

await mkdir(artifactRoot, { recursive: true });
await mkdir(join(tmpdir(), 'wllama-chromium'), { recursive: true });
const server = await createServer();
const results = [];
let runnerError = null;

try {
  for (const caseName of selectedCases) {
    if (receivedSignal) break;
    const modelCase = MODEL_CASES[caseName];
    if (!twoPhase) {
      results.push(await runCase(caseName, modelCase, server));
      continue;
    }

    const downloadResult = await runCase(caseName, modelCase, server, {
      artifactName: `${caseName}-download`,
      downloadOnly: true,
      keepBrowserProfile: true,
      reuseBrowserProfile: false,
    });
    results.push(downloadResult);
    if (downloadResult.status !== 'passed' || receivedSignal) {
      await rm(getBrowserProfile(caseName), {
        force: true,
        recursive: true,
      });
      continue;
    }

    // Close the downloader and flush its OPFS writes before committing the
    // model tensors in a fresh renderer. This lets Linux reclaim file cache.
    execFileSync('sync');
    results.push(
      await runCase(caseName, modelCase, server, {
        artifactName: `${caseName}-inference`,
        downloadOnly: false,
        keepBrowserProfile: false,
        reuseBrowserProfile: true,
      })
    );
  }
} catch (error) {
  runnerError = { message: error.message, stack: error.stack };
  writeLine(`Stress runner failed: ${error.stack || error.message}`);
} finally {
  try {
    await Promise.all(
      [...activeContexts].map((activeContext) =>
        activeContext.close().catch(() => {})
      )
    );
    await server.close();
  } finally {
    if (twoPhase) {
      await Promise.all(
        selectedCases.map((caseName) =>
          rm(getBrowserProfile(caseName), { force: true, recursive: true })
        )
      );
    }
  }
}

await writeFile(
  join(artifactRoot, 'summary.json'),
  `${JSON.stringify({ artifactRoot, results, runnerError }, null, 2)}\n`
);
writeLine(`Artifacts: ${artifactRoot}`);

if (receivedSignal) process.exitCode = receivedSignal === 'SIGINT' ? 130 : 143;
else if (runnerError) process.exitCode = 1;
else if (results.some((result) => result.status === 'failed'))
  process.exitCode = 1;
