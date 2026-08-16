/**
 * Tab-side proxy for running wllama inside a SharedWorker, so many tabs share one wasm module and one copy of the model in memory.
 *
 * ProxyToSharedWorker has the same surface as ProxyToWorker, but talks to a SharedWorker running SHARED_WORKER_SCOPE_CODE instead of spawning a dedicated Worker.
 *
 * Discovery is blob-based: localStorage keeps the blob URL of the running scope, a Web Lock serializes creation. The scope dies when the last tab closes.
 */

import type { GlueMsg } from './glue/messages';
import type { WllamaWorkerResources } from './worker';
import {
  LIBLLAMA_VERSION,
  LLAMA_CPP_WORKER_CODE,
  SHARED_WORKER_SCOPE_CODE,
  WLLAMA_EMSCRIPTEN_CODE,
} from './workers-code/generated';

interface Logger {
  debug: typeof console.debug;
  log: typeof console.log;
  warn: typeof console.warn;
  error: typeof console.error;
}

const SW_NAME = 'wllama';
const HB_INTERVAL_MS = 1000;
const ATTACH_TIMEOUT_MS = 3000;

const hashCode = (s: string): string => {
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  }
  return (h >>> 0).toString(36);
};

// tabs running different builds must not share one scope, so keys contain a version tag
const VERSION_TAG = `${LIBLLAMA_VERSION}-${hashCode(SHARED_WORKER_SCOPE_CODE)}`;
const REG_KEY = `wllama-sw-url::${VERSION_TAG}`;
const CREATE_LOCK = `wllama-sw-create::${VERSION_TAG}`;

export const isSharedWorkerSupported = (): boolean =>
  typeof SharedWorker !== 'undefined' &&
  typeof localStorage !== 'undefined' &&
  typeof navigator !== 'undefined' &&
  !!navigator.locks;

export type SharedWorkerStatus = 'uninit' | 'initializing' | 'ready' | 'error';

export interface SharedWorkerState {
  status: SharedWorkerStatus;
  // opaque data pushed by the tab that loaded the model
  snapshot: any;
  modelId: any;
}

interface PendingRpc {
  resolve: (value: any) => void;
  reject: (reason: any) => void;
}

export class ProxyToSharedWorker {
  resources: WllamaWorkerResources;
  suppressNativeLog: boolean;
  logger: Logger;

  private sw?: SharedWorker | undefined;
  private port?: MessagePort | undefined;
  private dead = false;
  private nextRpcId = 1;
  private pending: Map<number, PendingRpc> = new Map();
  private readyWaiters: PendingRpc[] = [];
  private hbTimer?: ReturnType<typeof setInterval>;

  constructor(
    resources: WllamaWorkerResources,
    suppressNativeLog: boolean,
    logger: Logger
  ) {
    this.resources = resources;
    this.suppressNativeLog = suppressNativeLog;
    this.logger = logger;
  }

  /**
   * Attach to the shared scope, create it if needed. Returns the scope state.
   */
  async connect(): Promise<SharedWorkerState> {
    if (this.port) {
      throw new Error('already connected');
    }
    const knownUrl = localStorage.getItem(REG_KEY);
    if (knownUrl) {
      try {
        await this.attach(knownUrl);
      } catch (e) {
        this.logger.debug('cannot attach to known shared worker, will create a new one', e);
      }
    }
    if (!this.port) {
      await navigator.locks.request(CREATE_LOCK, async () => {
        // another tab may have created the scope while we waited for the lock
        const url = localStorage.getItem(REG_KEY);
        if (url && url !== knownUrl) {
          try {
            await this.attach(url);
          } catch (e) {
            // fall through to create
          }
        }
        if (!this.port) {
          const newUrl = URL.createObjectURL(
            new Blob([SHARED_WORKER_SCOPE_CODE], { type: 'text/javascript' })
          );
          await this.attach(newUrl);
          localStorage.setItem(REG_KEY, newUrl);
        }
      });
    }
    this.hbTimer = setInterval(() => {
      this.rpc('hb').catch(() => {});
    }, HB_INTERVAL_MS);
    return await this.getState();
  }

  private attach(url: string): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const sw = new SharedWorker(url, SW_NAME);
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error('timeout waiting for shared worker'));
      }, ATTACH_TIMEOUT_MS);
      const cleanup = () => {
        clearTimeout(timer);
        sw.onerror = null;
      };
      sw.onerror = () => {
        cleanup();
        reject(new Error('shared worker script cannot be loaded'));
      };
      sw.port.onmessage = (e: MessageEvent) => {
        if (e.data?.evt === 'hello') {
          cleanup();
          this.sw = sw;
          this.port = sw.port;
          sw.port.onmessage = (ev: MessageEvent) => this.onRecvMsg(ev);
          resolve();
        }
      };
      sw.port.start();
    });
  }

  private onRecvMsg(e: MessageEvent) {
    const m = e.data;
    if (!m) return;
    if (m.evt) {
      if (m.evt === 'log') {
        const level = (m.level as keyof Logger) in this.logger ? (m.level as keyof Logger) : 'debug';
        this.logger[level](m.line);
      } else if (m.evt === 'ready') {
        for (const w of this.readyWaiters) w.resolve(null);
        this.readyWaiters = [];
      } else if (m.evt === 'dying') {
        this.shutdown(new Error('shared worker is closing: ' + m.reason));
      }
      return;
    }
    const p = this.pending.get(m.id);
    if (!p) return;
    this.pending.delete(m.id);
    if (m.err !== undefined) p.reject(new Error(m.err));
    else p.resolve(m.result);
  }

  private shutdown(reason: Error) {
    this.dead = true;
    if (this.hbTimer) clearInterval(this.hbTimer);
    for (const p of this.pending.values()) p.reject(reason);
    this.pending.clear();
    for (const w of this.readyWaiters) w.reject(reason);
    this.readyWaiters = [];
    this.port?.close();
    this.port = undefined;
  }

  private rpc(verb: string, payload?: any): Promise<any> {
    if (this.dead || !this.port) {
      return Promise.reject(new Error('not connected to shared worker'));
    }
    return new Promise((resolve, reject) => {
      const id = this.nextRpcId++;
      this.pending.set(id, { resolve, reject });
      this.port!.postMessage({ id, verb, payload });
    });
  }

  //////// state helpers (not part of ProxyToWorker surface) ////////

  getState(): Promise<SharedWorkerState> {
    return this.rpc('get-state');
  }

  setState(snapshot: any, modelId: any): Promise<void> {
    return this.rpc('set-state', { snapshot, modelId });
  }

  /**
   * Wait until another tab finishes loading the model
   */
  waitReady(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.readyWaiters.push({ resolve, reject });
      // in case ready happened between get-state and this call
      this.getState().then((s) => {
        if (s.status === 'ready') resolve(undefined);
      }, () => {});
    });
  }

  /**
   * Kill the shared scope for ALL tabs. Use wllamaExit() to only detach this tab.
   */
  async destroy(): Promise<void> {
    await this.rpc('exit').catch(() => {});
  }

  //////// same surface as ProxyToWorker ////////

  async moduleInit(ggufFiles: { name: string; blob: Blob }[]): Promise<void> {
    // the scope bundle does not contain the big generated code strings, send them along
    const resources: WllamaWorkerResources = {
      ...this.resources,
      jsPath: this.resources.jsPath ?? { code: WLLAMA_EMSCRIPTEN_CODE },
      llamaCppCode: this.resources.llamaCppCode ?? LLAMA_CPP_WORKER_CODE,
    };
    return await this.rpc('module-init', {
      resources,
      ggufFiles,
      suppressNativeLog: this.suppressNativeLog,
    });
  }

  async wllamaStart(): Promise<number> {
    return await this.rpc('start');
  }

  async wllamaAction<T extends GlueMsg>(name: string, body: GlueMsg): Promise<T> {
    // body is a plain glue object, it is glue-serialized inside the scope
    return await this.rpc('action', { name, body });
  }

  async wllamaExit(): Promise<void> {
    // detach this tab only, the scope keeps running for other tabs
    this.shutdown(new Error('proxy detached'));
  }

  async wllamaDebug(): Promise<any> {
    return await this.rpc('debug');
  }
}
