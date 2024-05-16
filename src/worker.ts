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
 */
const MEMFS_PATCH_TO_HEAPFS = `
const fileToPtr = {};

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
    if (fileToPtr[name]) {
      const f = fileToPtr[name];
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
    if (fileToPtr[name]) {
      const f = fileToPtr[name];
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

// Add new file to wllama heapfs
const heapfsWriteFile = async (name, buf) => {
  const m = wModule;
  const ptr = m.mmapAlloc(buf.byteLength);
  m.HEAPU8.set(new Uint8Array(buf), ptr);
  fileToPtr[name] = {
    ptr: ptr,
    size: buf.byteLength,
  };
};
`;

//////////////////////////////////////////////////
//////////////////////////////////////////////////

const WORKER_UTILS = `
// send message back to main thread
const msg = (data) => postMessage(data);

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
      msg({ verb: 'console.warn', args: [text] });
    },
    locateFile: function (filename, basePath) {
      const p = pathConfig[filename];
      const truncate = (str) => str.length > 128 ? \`\${str.substr(0, 128)}...\` : str;
      msg({ verb: 'console.log', args: [\`Loading "\${filename}" from "\${truncate(p)}"\`] });
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
    const argPathConfig     = args[0];
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

  if (verb === 'module.upload') {
    const argFilename = args[0]; // file name
    const argBuffer   = args[1]; // buffer for file data
    try {
      // create blank file
      const empty = new ArrayBuffer(0);
      wModule['FS_createDataFile']('/models', argFilename, empty, true, true, true);
      // write data to heap
      await heapfsWriteFile(argFilename, argBuffer);
      msg({ callbackId, result: true });
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
  verb: 'module.init'
  | 'module.upload'
  | 'wllama.start'
  | 'wllama.action'
  | 'wllama.exit'
  | 'wllama.debug',
  args: any[],
  callbackId: number,
};

interface Task { resolve: any, reject: any, param: TaskParam };

export class ProxyToWorker {
  taskQueue: Task[] = [];
  taskId: number = 1;
  resultQueue: Task[] = [];
  busy = false; // is the work loop is running?
  worker?: Worker;
  pathConfig: any;
  multiThread: boolean;
  nbThread: number;

  constructor(pathConfig: any, nbThread: number = 1) {
    this.pathConfig = pathConfig;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 1;
  }

  async moduleInit(ggufBuffers: ArrayBuffer[]): Promise<void> {
    if (!this.pathConfig['wllama.js']) {
      throw new Error('"single-thread/wllama.js" or "multi-thread/wllama.js" is missing from pathConfig');
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
    const workerURL = window.URL.createObjectURL(new Blob([completeCode], { type: 'text/javascript' }));
    this.worker = new Worker(workerURL);
    this.worker.onmessage = this.onRecvMsg.bind(this);
    this.worker.onerror = console.error;

    const res = await this.pushTask({
      verb: 'module.init',
      args: [
        this.pathConfig,
        this.nbThread,
      ],
      callbackId: this.taskId++,
    });

    // copy buffer to worker
    for (let i = 0; i < ggufBuffers.length; i++) {
      await this.pushTask({
        verb: 'module.upload',
        args: [
          ggufBuffers.length === 1
            ? 'model.gguf'
            : `model-${padDigits(i + 1, 5)}-of-${padDigits(ggufBuffers.length, 5)}.gguf`,
          new Uint8Array(ggufBuffers[i]),
        ],
        callbackId: this.taskId++,
      });
      freeBuffer(ggufBuffers[i]);
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

  async wllamaAction(name: string, body: any): Promise<any> {
    const result = await this.pushTask({
      verb: 'wllama.action',
      args: [name, JSON.stringify(body)],
      callbackId: this.taskId++,
    });
    const parsedResult = this.parseResult(result);
    return parsedResult;
  }

  async wllamaExit(): Promise<{ success: boolean }> {
    const result = await this.pushTask({
      verb: 'wllama.exit',
      args: [],
      callbackId: this.taskId++,
    });
    const parsedResult = this.parseResult(result);
    return parsedResult;
  }

  async wllamaDebug(): Promise<any> {
    const result = await this.pushTask({
      verb: 'wllama.debug',
      args: [],
      callbackId: this.taskId++,
    });
    return JSON.parse(result);
  }

  private parseResult(result: any): any {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult['__exception']) {
      throw new Error(parsedResult['__exception']);
    }
    return parsedResult;
  }

  private pushTask(param: TaskParam) {
    return new Promise<any>((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param });
      this.runTaskLoop();
    });
  }

  private async runTaskLoop() {
    if (this.busy) {
      return; // another loop is already running
    }
    this.busy = true;
    while (true) {
      const task = this.taskQueue.shift();
      if (!task) break; // no more tasks
      this.resultQueue.push(task);
      this.worker!!.postMessage(task.param);
    }
    this.busy = false;
  }

  private onRecvMsg(e: MessageEvent<any>) {
    if (!e.data) return; // ignore
    const { verb, args } = e.data;
    if (verb && verb.startsWith('console.')) {
      if (verb.endsWith('log')) console.log(...args);
      if (verb.endsWith('warn')) console.warn(...args);
      if (verb.endsWith('error')) console.error(...args);
      return;
    } else if (verb === 'signal.abort') {
      this.abort(args[0]);
    }

    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(t => t.param.callbackId === callbackId);
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        console.error(`Cannot find waiting task with callbackId = ${callbackId}`);
      }
    }
  }

  private abort(text: string) {
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(new Error(`Received abort signal from llama.cpp; Message: ${text || '(empty)'}`));
    }
  }
}

/**
 * Utility functions
 */

// Free ArrayBuffer by resizing them to 0. This is needed because sometimes we run into OOM issue.
function freeBuffer(buf: ArrayBuffer) {
  // @ts-ignore
  if (ArrayBuffer.prototype.transfer) {
    // @ts-ignore
    buf.transfer(0);
    // @ts-ignore
  } else if (ArrayBuffer.prototype.resize && buf.resizable) {
    // @ts-ignore
    buf.resize(0);
  } else {
    console.warn('Cannot free buffer. You may run into out-of-memory issue.');
  }
}

// Zero-padding numbers
function padDigits(number: number, digits: number) {
  return Array(Math.max(digits - String(number).length + 1, 0)).join('0') + number;
}
