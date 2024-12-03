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

import { createWorker, isSafariMobile } from './utils';
import {
  LLAMA_CPP_WORKER_CODE,
  WLLAMA_MULTI_THREAD_CODE,
  WLLAMA_MULTI_THREAD_WORKER_CODE,
  WLLAMA_SINGLE_THREAD_CODE,
} from './workers-code/generated';

interface Logger {
  debug: typeof console.debug;
  log: typeof console.log;
  warn: typeof console.warn;
  error: typeof console.error;
}

interface TaskParam {
  verb:
    | 'module.init'
    | 'fs.alloc'
    | 'fs.write'
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
  buffers?: ArrayBuffer[];
}

export class ProxyToWorker {
  logger: Logger;
  suppressNativeLog: boolean;
  taskQueue: Task[] = [];
  taskId: number = 1;
  resultQueue: Task[] = [];
  busy = false; // is the work loop is running?
  worker?: Worker;
  pathConfig: any;
  multiThread: boolean;
  nbThread: number;

  constructor(
    pathConfig: any,
    nbThread: number = 1,
    suppressNativeLog: boolean,
    logger: Logger
  ) {
    this.pathConfig = pathConfig;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 1;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
  }

  async moduleInit(ggufFiles: { name: string; blob: Blob }[]): Promise<void> {
    if (!this.pathConfig['wllama.wasm']) {
      throw new Error('"single-thread/wllama.wasm" is missing from pathConfig');
    }
    let moduleCode = this.multiThread
      ? WLLAMA_MULTI_THREAD_CODE
      : WLLAMA_SINGLE_THREAD_CODE;
    moduleCode = moduleCode.replace('var Module', 'var ___Module');
    const runOptions = {
      pathConfig: this.pathConfig,
      nbThread: this.nbThread,
    };
    const completeCode: string = [
      this.multiThread
        ? `const WLLAMA_MULTI_THREAD_WORKER_CODE = ${JSON.stringify(WLLAMA_MULTI_THREAD_WORKER_CODE)};`
        : '// single-thread build',
      `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
      `function wModuleInit() { ${moduleCode}; return Module; }`,
      LLAMA_CPP_WORKER_CODE,
    ].join(';\n\n');
    this.worker = createWorker(completeCode);
    this.worker.onmessage = this.onRecvMsg.bind(this);
    this.worker.onerror = this.logger.error;

    const res = await this.pushTask({
      verb: 'module.init',
      args: [new Blob([moduleCode], { type: 'text/javascript' })],
      callbackId: this.taskId++,
    });

    // allocate all files
    const nativeFiles: ({ id: number } & (typeof ggufFiles)[number])[] = [];
    for (const file of ggufFiles) {
      const id = await this.fileAlloc(file.name, file.blob.size);
      nativeFiles.push({ id, ...file });
    }

    // stream files
    await Promise.all(
      nativeFiles.map((file) => {
        return this.fileWrite(file.id, file.blob);
      })
    );

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

  async wllamaAction(name: string, body: any): Promise<any> {
    const result = await this.pushTask({
      verb: 'wllama.action',
      args: [name, JSON.stringify(body)],
      callbackId: this.taskId++,
    });
    const parsedResult = this.parseResult(result);
    return parsedResult;
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
  private async fileAlloc(fileName: string, size: number): Promise<number> {
    const result = await this.pushTask({
      verb: 'fs.alloc',
      args: [fileName, size],
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

  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   */
  private parseResult(result: any): any {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult['__exception']) {
      throw new Error(parsedResult['__exception']);
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
