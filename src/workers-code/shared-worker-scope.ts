/**
 * This code runs inside a SharedWorker. It hosts the real ProxyToWorker and the wasm module, so one model instance can serve many tabs.
 *
 * Messages between tab <==> this scope:
 * - Send direction: { id, verb, payload }
 * - Result direction: { id, result } or { id, err }
 * - Broadcast events (no id): { evt: 'log' | 'ready' | 'dying', ... }
 *
 * Browsers do not allow "new Worker()" in this scope, the wasm worker code runs here via importScripts (see makeInScopeWorker). SharedArrayBuffer is also not available, so nbThread is always 0.
 *
 * Must NOT import workers-code/generated (it would double the bundle size), the tab sends the code strings via resources. See scripts/build_shared_worker_scope.mjs.
 */

import { ProxyToWorker, type WllamaWorkerResources } from '../worker';

// this file is compiled with lib "dom", worker-only globals are declared by hand here
declare const self: Record<string, any>;
declare function importScripts(...urls: string[]): void;

const HB_TIMEOUT_MS = 6000;
const SWEEP_INTERVAL_MS = 2000;

// llama-cpp.js expects dedicated worker globals: it calls postMessage() and assigns onmessage. We provide/capture these around importScripts and bridge them to a fake Worker object.
// Deliveries go through a microtask to keep the same async ordering as a real Worker.
const makeInScopeWorker = (code: string | Blob): Worker => {
  let handler: ((e: { data: any }) => void) | null = null;
  const fake = {
    onmessage: null as ((e: { data: any }) => void) | null,
    onerror: null as ((e: any) => void) | null,
    postMessage: (data: any) => {
      Promise.resolve().then(() => handler && handler({ data }));
    },
    terminate: () => {},
  };
  self.postMessage = (data: any) => {
    Promise.resolve().then(() => fake.onmessage && fake.onmessage({ data }));
  };
  const url = URL.createObjectURL(
    new Blob([code], { type: 'text/javascript' })
  );
  importScripts(url);
  handler = self.onmessage; // assigned by llama-cpp.js at eval time
  self.onmessage = null;
  return fake as any as Worker;
};

//////////////////////////////////////////////////////////////
// STATE
//////////////////////////////////////////////////////////////

type Status = 'uninit' | 'initializing' | 'ready' | 'error';

let proxy: ProxyToWorker | null = null;
let status: Status = 'uninit';
let lastError: string | null = null;
let initializerPortId = -1;

// opaque data pushed by the tab that loaded the model, sent back to late-joining tabs so they can skip loading
let snapshot: any = null;
let modelId: any = null;

interface TabPort {
  port: MessagePort;
  lastSeen: number;
}
let nextPortId = 1;
const ports = new Map<number, TabPort>();

const broadcast = (msg: any) => {
  for (const t of ports.values()) {
    try {
      t.port.postMessage(msg);
    } catch (e) {
      // port is dead, sweep will remove it
    }
  }
};

const logs: string[] = [];
const record = (level: string, args: any[]) => {
  const line = `[${level}] ` + args.map((a) => String(a)).join(' ');
  logs.push(line);
  if (logs.length > 200) logs.shift();
  broadcast({ evt: 'log', level, line });
};
const logger = {
  debug: (...args: any[]) => record('debug', args),
  log: (...args: any[]) => record('log', args),
  warn: (...args: any[]) => record('warn', args),
  error: (...args: any[]) => record('error', args),
};

//////////////////////////////////////////////////////////////
// LIFETIME
//////////////////////////////////////////////////////////////

// a SharedWorker gets no event when a tab closes, so tabs send 'hb' and we prune silent ports here
setInterval(() => {
  const now = Date.now();
  for (const [portId, t] of ports) {
    if (now - t.lastSeen > HB_TIMEOUT_MS) {
      ports.delete(portId);
      // initializer tab died mid-init, module state is unknown, kill the scope so the next tab starts clean
      if (status === 'initializing' && portId === initializerPortId) {
        die('initializer tab died during init');
      }
    }
  }
}, SWEEP_INTERVAL_MS);

const die = (reason: string) => {
  logger.warn('shared worker scope closing: ' + reason);
  broadcast({ evt: 'dying', reason });
  setTimeout(() => self.close(), 10);
};

//////////////////////////////////////////////////////////////
// RPC HANDLERS
//////////////////////////////////////////////////////////////

interface ModuleInitPayload {
  resources: WllamaWorkerResources;
  ggufFiles: { name: string; blob: Blob }[];
  suppressNativeLog: boolean;
}

const handleModuleInit = async (portId: number, p: ModuleInitPayload) => {
  if (status !== 'uninit') {
    throw new Error('cannot init, current status: ' + status);
  }
  if (!(p.resources.jsPath as { code: string })?.code || !p.resources.llamaCppCode) {
    throw new Error('resources must contain jsPath.code and llamaCppCode (generated code is not bundled in this scope)');
  }
  status = 'initializing';
  initializerPortId = portId;
  try {
    proxy = new ProxyToWorker(p.resources, 0, p.suppressNativeLog, logger, makeInScopeWorker);
    await proxy.moduleInit(p.ggufFiles);
    return null;
  } catch (e: any) {
    status = 'error';
    lastError = String(e?.message || e);
    throw e;
  }
};

const requireProxy = (): ProxyToWorker => {
  if (!proxy || status === 'uninit' || status === 'error') {
    throw new Error('module not initialized, current status: ' + status + (lastError ? ', last error: ' + lastError : ''));
  }
  return proxy;
};

const handleVerb = async (portId: number, verb: string, payload: any): Promise<any> => {
  switch (verb) {
    case 'hb':
      return null;
    case 'status':
      return { status, lastError, modelId, nbTabs: ports.size, logs: logs.slice(-50) };
    case 'module-init':
      return await handleModuleInit(portId, payload);
    case 'start':
      return await requireProxy().wllamaStart();
    case 'action':
      return await requireProxy().wllamaAction(payload.name, payload.body);
    case 'debug':
      return await requireProxy().wllamaDebug();
    case 'get-state':
      return { status, snapshot, modelId };
    case 'set-state':
      snapshot = payload.snapshot;
      modelId = payload.modelId;
      if (status === 'initializing') {
        status = 'ready';
        initializerPortId = -1;
        broadcast({ evt: 'ready' });
      }
      return null;
    case 'exit':
      // tear down the whole instance, other tabs get 'dying'
      try {
        await requireProxy().wllamaExit();
      } finally {
        die('exit requested');
      }
      return null;
    case 'reset':
      // hard reset without touching the wasm
      die('reset requested');
      return null;
    default:
      throw new Error('unknown verb: ' + verb);
  }
};

self.onconnect = (e: MessageEvent) => {
  const port = e.ports[0];
  const portId = nextPortId++;
  ports.set(portId, { port, lastSeen: Date.now() });
  port.onmessage = async (ev: MessageEvent) => {
    const { id, verb, payload } = ev.data;
    const t = ports.get(portId);
    if (t) t.lastSeen = Date.now();
    try {
      const result = await handleVerb(portId, verb, payload);
      port.postMessage({ id, result });
    } catch (err: any) {
      port.postMessage({ id, err: String(err?.message || err) });
    }
  };
  port.postMessage({ evt: 'hello', portId });
};
