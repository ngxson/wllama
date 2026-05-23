/**
 * Module code will be copied into worker.
 *
 * Messages between main <==> worker:
 *
 * From main thread to worker:
 * - Send direction: { verb, args, callbackId }
 * - Result direction: { callbackId, result } or { callbackId, err }
 *
 * Signal from worker to main:
 * - Unidirection: { verb, args }
 */

import { glueDeserialize, glueSerialize } from './glue/glue';
import type { GlueMsg } from './glue/messages';
import {
  canUseAsyncFileRead,
  createWorker,
  isSafariMobile,
  isString,
} from './utils';
import {
  LLAMA_CPP_WORKER_CODE,
  WLLAMA_EMSCRIPTEN_CODE,
} from './workers-code/generated';

interface Logger {
  debug: typeof console.debug;
  log: typeof console.log;
  warn: typeof console.warn;
  error: typeof console.error;
}

const FILE_READ_REQ_EVENT = 'fs.read_req';

interface TaskParam {
  verb:
    | 'module.init'
    | 'fs.alloc'
    | 'fs.write'
    | 'fs.read_res'
    | 'wllama.start'
    | 'wllama.action'
    | 'wllama.exit'
    | 'wllama.debug';
  args: any[];
  callbackId: number;
}

interface Task {
  resolve: any;
  reject: any;
  param: TaskParam;
  buffers?: ArrayBuffer[] | undefined;
}

const JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;

export interface WllamaWorkerResources {
  wasmPath: string;
  // if jsPath is not provided, use WLLAMA_EMSCRIPTEN_CODE
  jsPath?: string | { code: string } | undefined;
  // in compat mode, mem64 must be disabled
  compat: boolean;
}

export class ProxyToWorker {
  resources: WllamaWorkerResources;
  logger: Logger;
  suppressNativeLog: boolean;
  taskQueue: Task[] = [];
  taskId: number = 1;
  resultQueue: Task[] = [];
  busy = false; // is the work loop is running?
  worker?: Worker | undefined;
  multiThread: boolean;
  nbThread: number;
  useAsyncFile: boolean;
  fileBlobs: Map<string, Blob> = new Map(); // filename -> Blob for async reads

  constructor(
    resources: WllamaWorkerResources,
    nbThread: number,
    suppressNativeLog: boolean,
    logger: Logger
  ) {
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }

  async getModuleCode(): Promise<string> {
    if (!this.resources.jsPath) {
      if (this.resources.compat) {
        throw new Error(
          'compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat.'
        );
      }
      return WLLAMA_EMSCRIPTEN_CODE;
    } else if ((this.resources.jsPath as { code: string }).code) {
      return (this.resources.jsPath as { code: string }).code;
    } else if (isString(this.resources.jsPath)) {
      const response = await fetch(this.resources.jsPath as string);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch worker code from ${this.resources.jsPath}`
        );
      }
      return await response.text();
    } else {
      throw new Error('No JS code provided for worker');
    }
  }

  async moduleInit(ggufFiles: { name: string; blob: Blob }[]): Promise<void> {
    let moduleCode = JSPI_STUB + (await this.getModuleCode());
    let mainModuleCode = moduleCode.replace('var Module', 'var ___Module');
    const runOptions = {
      pathConfig: {
        'wllama.wasm': this.resources.wasmPath,
      },
      nbThread: this.nbThread,
      compat: this.resources.compat,
    };
    const completeCode: string = [
      `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
      `function wModuleInit() { ${mainModuleCode}; return Module; }`,
      LLAMA_CPP_WORKER_CODE,
    ].join(';\n\n');
    this.worker = createWorker(completeCode);
    this.worker.onmessage = this.onRecvMsg.bind(this);
    this.worker.onerror = this.logger.error;

    const res = await this.pushTask({
      verb: 'module.init',
      args: [
        new Blob([moduleCode], { type: 'text/javascript' }),
        this.useAsyncFile,
      ],
      callbackId: this.taskId++,
    });

    // allocate all files
    const nativeFiles: ({ id: number } & (typeof ggufFiles)[number])[] = [];
    for (const file of ggufFiles) {
      const needAllocBuffer = !this.useAsyncFile; // only alloc if mmap is used
      const id = await this.fileAlloc(
        file.name,
        file.blob.size,
        needAllocBuffer
      );
      nativeFiles.push({ id, ...file });
      if (this.useAsyncFile) {
        this.fileBlobs.set(file.name, file.blob);
      }
    }

    // stream files (only used in non async - mmap mode)
    if (!this.useAsyncFile) {
      await Promise.all(
        nativeFiles.map((file) => {
          return this.fileWrite(file.id, file.blob);
        })
      );
    }

    return res;
  }

  async wllamaStart(): Promise<number> {
    const result = await this.pushTask({
      verb: 'wllama.start',
      args: [],
      callbackId: this.taskId++,
    });
    const parsedResult = this.parseResult(result);
    return parsedResult;
  }

  async wllamaAction<T extends GlueMsg>(
    name: string,
    body: GlueMsg
  ): Promise<T> {
    // console.debug(`wllamaAction: ${name}`, body);
    const encodedMsg = glueSerialize(body);
    const result = await this.pushTask({
      verb: 'wllama.action',
      args: [name, encodedMsg],
      callbackId: this.taskId++,
    });
    const parsedResult = glueDeserialize(result);
    return parsedResult as T;
  }

  async wllamaExit(): Promise<void> {
    if (this.worker) {
      const result = await this.pushTask({
        verb: 'wllama.exit',
        args: [],
        callbackId: this.taskId++,
      });
      this.parseResult(result); // only check for exceptions
      this.worker.terminate();
    }
  }

  async wllamaDebug(): Promise<any> {
    const result = await this.pushTask({
      verb: 'wllama.debug',
      args: [],
      callbackId: this.taskId++,
    });
    return JSON.parse(result);
  }

  ///////////////////////////////////////

  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  private async fileAlloc(
    fileName: string,
    size: number,
    allocBuffer: boolean
  ): Promise<number> {
    const result = await this.pushTask({
      verb: 'fs.alloc',
      args: [fileName, size, allocBuffer],
      callbackId: this.taskId++,
    });
    return result.fileId;
  }

  /**
   * Write a Blob to heapfs
   */
  private async fileWrite(fileId: number, blob: Blob): Promise<void> {
    const reader = blob.stream().getReader();
    let offset = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const size = value.byteLength;
      await this.pushTask(
        {
          verb: 'fs.write',
          args: [fileId, value, offset],
          callbackId: this.taskId++,
        },
        // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
        [value.buffer]
      );
      offset += size;
    }
  }

  private async fileReadResponse(
    name: string,
    offset: number,
    size: number
  ): Promise<void> {
    try {
      const blob = this.fileBlobs.get(name);
      if (!blob) {
        throw new Error(`blob not found for name="${name}"`);
      }
      const chunk = blob.slice(offset, offset + size);
      const buffer = await chunk.arrayBuffer();
      this.worker!!.postMessage(
        { verb: 'fs.read_res', args: [buffer] },
        { transfer: [buffer] }
      );
    } catch (err) {
      this.logger.error('fileReadResponse failed, terminating worker:', err);
      this.worker?.terminate();
      this.worker = undefined;
      this.abort(`File read failed: ${err}`);
    }
  }

  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  private parseResult(result: any): any {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult['error']) {
      throw new Error('Unknown error, please see console.log');
    }
    return parsedResult;
  }

  /**
   * Push a new task to taskQueue
   */
  private pushTask(param: TaskParam, buffers?: ArrayBuffer[]) {
    return new Promise<any>((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }

  /**
   * Main loop for processing tasks
   */
  private async runTaskLoop() {
    if (this.busy) {
      return; // another loop is already running
    }
    this.busy = true;
    while (true) {
      const task = this.taskQueue.shift();
      if (!task) break; // no more tasks
      this.resultQueue.push(task);
      // TODO @ngxson : Safari mobile doesn't support transferable ArrayBuffer
      this.worker!!.postMessage(
        task.param,
        isSafariMobile()
          ? undefined
          : {
              transfer: task.buffers ?? [],
            }
      );
    }
    this.busy = false;
  }

  /**
   * Handle messages from worker
   */
  private onRecvMsg(e: MessageEvent<any>) {
    if (!e.data) return; // ignore
    const { verb, args } = e.data;
    if (verb && verb.startsWith('console.')) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith('debug')) this.logger.debug(...args);
      if (verb.endsWith('log')) this.logger.log(...args);
      if (verb.endsWith('warn')) this.logger.warn(...args);
      if (verb.endsWith('error')) this.logger.error(...args);
      return;
    } else if (verb === 'signal.abort') {
      this.abort(args[0]);
    }

    // handle fs.read_req signal from wasm (JSPI-suspended worker)
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args as [string, number, number];
      this.fileReadResponse(name, offset, size).catch(() => {}); // errors handled inside
      return;
    }

    // handle task result
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }

  private abort(text: string) {
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(
        new Error(
          `Received abort signal from llama.cpp; Message: ${text || '(empty)'}`
        )
      );
    }
  }
}
