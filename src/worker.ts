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

import { isSafariMobile } from './utils';

interface Logger {
  debug: typeof console.debug;
  log: typeof console.log;
  warn: typeof console.warn;
  error: typeof console.error;
}

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/**
 * By default, emscripten uses memfs. The way it works is by
 * allocating new Uint8Array in javascript heap. This is not good
 * because it requires files to be copied to wasm heap each time
 * a file is read.
 *
 * HeapFS is an alternative, which resolves this problem by
 * allocating space for file directly inside wasm heap. This
 * allows us to mmap without doing any copy.
 *
 * For llama.cpp, this is great because we use MAP_SHARED
 *
 * Ref: https://github.com/ngxson/wllama/pull/39
 * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js
 *
 * Note 29/05/2024 @ngxson
 * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)
 * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c
 */
const MEMFS_PATCH_TO_HEAPFS = `
const fsNameToFile = {};  // map Name => File
const fsIdToFile = {};    // map ID => File
let currFileId = 0;

// Patch and redirect memfs calls to wllama
const patchMEMFS = () => {
  const m = wModule;
  // save functions
  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;
  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;
  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;
  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;
  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;
  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;

  const patchStream = (stream) => {
    const name = stream.node.name;
    if (fsNameToFile[name]) {
      const f = fsNameToFile[name];
      stream.node.contents = m.HEAPU8.subarray(f.ptr, f.ptr + f.size);
      stream.node.usedBytes = f.size;
    }
  };

  // replace "read" functions
  m.MEMFS.stream_ops.read = function (stream, buffer, offset, length, position) {
    patchStream(stream);
    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);
  };
  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;

  // replace "llseek" functions
  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {
    patchStream(stream);
    return m.MEMFS.stream_ops._llseek(stream, offset, whence);
  };
  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;

  // replace "mmap" functions
  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {
    patchStream(stream);
    const name = stream.node.name;
    if (fsNameToFile[name]) {
      const f = fsNameToFile[name];
      return {
        ptr: f.ptr + position,
        allocated: false,
      };
    } else {
      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);
    }
  };
  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;

  // mount FS
  m.FS.mkdir('/models');
  m.FS.mount(m.MEMFS, { root: '.' }, '/models');
};

// Allocate a new file in wllama heapfs, returns file ID
const heapfsAlloc = (name, size) => {
  if (size < 1) {
    throw new Error('File size must be bigger than 0');
  }
  const m = wModule;
  const ptr = m.mmapAlloc(size);
  const file = {
    ptr: ptr,
    size: size,
    id: currFileId++,
  };
  fsIdToFile[file.id] = file;
  fsNameToFile[name] = file;
  return file.id;
};

// Add new file to wllama heapfs, return number of written bytes
const heapfsWrite = (id, buffer, offset) => {
  const m = wModule;
  if (fsIdToFile[id]) {
    const { ptr, size } = fsIdToFile[id];
    const afterWriteByte = offset + buffer.byteLength;
    if (afterWriteByte > size) {
      throw new Error(\`File ID \${id} write out of bound, afterWriteByte = \${afterWriteByte} while size = \${size}\`);
    }
    m.HEAPU8.set(buffer, ptr + offset);
    return buffer.byteLength;
  } else {
    throw new Error(\`File ID \${id} not found in heapfs\`);
  }
};
`;

//////////////////////////////////////////////////
//////////////////////////////////////////////////

const WORKER_UTILS = `
// send message back to main thread
const msg = (data) => postMessage(data);

// Convert CPP log into JS log
const cppLogToJSLog = (line) => {
  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);
  return !!matched
    ? {
      level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),
      text: matched[2],
    }
    : { level: 'log', text: line };
};

// Get module config that forwards stdout/err to main thread
const getWModuleConfig = (pathConfig, pthreadPoolSize) => {
  if (!pathConfig['wllama.js']) {
    throw new Error('"wllama.js" is missing in pathConfig');
  }
  return {
    noInitialRun: true,
    print: function (text) {
      if (arguments.length > 1) text = Array.prototype.slice.call(arguments).join(' ');
      msg({ verb: 'console.log', args: [text] });
    },
    printErr: function (text) {
      if (arguments.length > 1) text = Array.prototype.slice.call(arguments).join(' ');
      const logLine = cppLogToJSLog(text);
      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });
    },
    locateFile: function (filename, basePath) {
      const p = pathConfig[filename];
      const truncate = (str) => str.length > 128 ? \`\${str.substr(0, 128)}...\` : str;
      msg({ verb: 'console.debug', args: [\`Loading "\${filename}" from "\${truncate(p)}"\`] });
      return p;
    },
    mainScriptUrlOrBlob: pathConfig['wllama.js'],
    pthreadPoolSize,
    wasmMemory: pthreadPoolSize > 1 ? getWasmMemory() : null,
    onAbort: function (text) {
      msg({ verb: 'signal.abort', args: [text] });
    },
  };
};

// Get the memory to be used by wasm. (Only used in multi-thread mode)
// Because we have a weird OOM issue on iOS, we need to try some values
// See: https://github.com/emscripten-core/emscripten/issues/19144
//      https://github.com/godotengine/godot/issues/70621
const getWasmMemory = () => {
  let minBytes = 128 * 1024 * 1024;
  let maxBytes = 4096 * 1024 * 1024;
  let stepBytes = 128 * 1024 * 1024;
  while (maxBytes > minBytes) {
    try {
      const wasmMemory = new WebAssembly.Memory({
        initial: minBytes / 65536,
        maximum: maxBytes / 65536,
        shared: true,
      });
      return wasmMemory;
    } catch (e) {
      maxBytes -= stepBytes;
      continue; // retry
    }
  }
  throw new Error('Cannot allocate WebAssembly.Memory');
};
`;

//////////////////////////////////////////////////
//////////////////////////////////////////////////

const WORKER_CODE = `
// Start the main llama.cpp
let wModule;
let wllamaStart;
let wllamaAction;
let wllamaExit;
let wllamaDebug;

${WORKER_UTILS}

${MEMFS_PATCH_TO_HEAPFS}

const callWrapper = (name, ret, args) => {
  const fn = wModule.cwrap(name, ret, args);
  return async (action, req) => {
    let result;
    try {
      if (args.length === 2) {
        result = await fn(action, req);
      } else {
        result = fn();
      }
    } catch (ex) {
      console.error(ex);
      throw ex;
    }
    return result;
  };
}

onmessage = async (e) => {
  if (!e.data) return;
  const { verb, args, callbackId } = e.data;

  if (!callbackId) {
    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });
    return;
  }

  if (verb === 'module.init') {
    const argPathConfig      = args[0];
    const argPThreadPoolSize = args[1];
    try {
      const Module = ModuleWrapper();
      wModule = await Module(getWModuleConfig(
        argPathConfig,
        argPThreadPoolSize,
      ));

      // init FS
      patchMEMFS();

      // init cwrap
      wllamaStart  = callWrapper('wllama_start' , 'string', []);
      wllamaAction = callWrapper('wllama_action', 'string', ['string', 'string']);
      wllamaExit   = callWrapper('wllama_exit'  , 'string', []);
      wllamaDebug  = callWrapper('wllama_debug' , 'string', []);
      msg({ callbackId, result: null });

    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'fs.alloc') {
    const argFilename = args[0];
    const argSize     = args[1];
    try {
      // create blank file
      const emptyBuffer = new ArrayBuffer(0);
      wModule['FS_createDataFile']('/models', argFilename, emptyBuffer, true, true, true);
      // alloc data on heap
      const fileId = heapfsAlloc(argFilename, argSize);
      msg({ callbackId, result: { fileId } });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'fs.write') {
    const argFileId = args[0];
    const argBuffer = args[1];
    const argOffset = args[2];
    try {
      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);
      msg({ callbackId, result: { writtenBytes } });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'wllama.start') {
    try {
      const result = await wllamaStart();
      msg({ callbackId, result });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'wllama.action') {
    const argAction = args[0];
    const argBody = args[1];
    try {
      const result = await wllamaAction(argAction, argBody);
      msg({ callbackId, result });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'wllama.exit') {
    try {
      const result = await wllamaExit();
      msg({ callbackId, result });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'wllama.debug') {
    try {
      const result = await wllamaDebug();
      msg({ callbackId, result });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }
};
`;

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
    if (!this.pathConfig['wllama.js']) {
      throw new Error(
        '"single-thread/wllama.js" or "multi-thread/wllama.js" is missing from pathConfig'
      );
    }
    const Module = await import(this.pathConfig['wllama.js']);
    let moduleCode = Module.default.toString();
    // monkey-patch: remove all "import.meta"
    // FIXME: this monkey-patch will remove support for nodejs
    moduleCode = moduleCode.replace(/import\.meta/g, 'importMeta');
    const completeCode = [
      'const importMeta = {}',
      `function ModuleWrapper() {
        const _scriptDir = ${JSON.stringify(window.location.href)};
        return ${moduleCode};
      }`,
      WORKER_CODE,
    ].join(';\n\n');
    // https://stackoverflow.com/questions/5408406/web-workers-without-a-separate-javascript-file
    const workerURL = window.URL.createObjectURL(
      new Blob([completeCode], { type: 'text/javascript' })
    );
    this.worker = new Worker(workerURL);
    this.worker.onmessage = this.onRecvMsg.bind(this);
    this.worker.onerror = this.logger.error;

    const res = await this.pushTask({
      verb: 'module.init',
      args: [this.pathConfig, this.nbThread],
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
