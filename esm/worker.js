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
const WORKER_CODE = `
// send message back to main thread
const msg = (data) => postMessage(data);

// Get module config that forwards stdout/err to main thread
const getWModuleConfig = (pathConfig, pthreadPoolSize) => {
  if (!pathConfig['wllama.js']) {
    throw new Error('"wllama.js" is missing in pathConfig');
  }
  return {
    noInitialRun: true,
    print: function(text) {
      if (arguments.length > 1) text = Array.prototype.slice.call(arguments).join(' ');
      msg({ verb: 'console.log', args: [text] });
    },
    printErr: function(text) {
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

// Start the main llama.cpp
let wModule;
let wllamaStart;
let wllamaAction;
let wllamaExit;

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
      wModule['FS_createPath']('/', 'models', true, true);

      // init cwrap
      wllamaStart  = callWrapper('wllama_start' , 'string', []);
      wllamaAction = callWrapper('wllama_action', 'string', ['string', 'string']);
      wllamaExit   = callWrapper('wllama_exit'  , 'string', []);
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
      wModule['FS_createDataFile']('/models', argFilename, argBuffer, true, true, true);
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
};
`;
;
;
export class ProxyToWorker {
    taskQueue = [];
    taskId = 1;
    resultQueue = [];
    busy = false; // is the work loop is running?
    worker;
    pathConfig;
    multiThread;
    nbThread;
    constructor(pathConfig, nbThread = 1) {
        this.pathConfig = pathConfig;
        this.nbThread = nbThread;
        this.multiThread = nbThread > 1;
    }
    async moduleInit(ggufBuffers) {
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
    async wllamaStart() {
        const result = await this.pushTask({
            verb: 'wllama.start',
            args: [],
            callbackId: this.taskId++,
        });
        const parsedResult = this.parseResult(result);
        return parsedResult;
    }
    async wllamaAction(name, body) {
        const result = await this.pushTask({
            verb: 'wllama.action',
            args: [name, JSON.stringify(body)],
            callbackId: this.taskId++,
        });
        const parsedResult = this.parseResult(result);
        return parsedResult;
    }
    async wllamaExit() {
        const result = await this.pushTask({
            verb: 'wllama.exit',
            args: [],
            callbackId: this.taskId++,
        });
        const parsedResult = this.parseResult(result);
        return parsedResult;
    }
    parseResult(result) {
        const parsedResult = JSON.parse(result);
        if (parsedResult && parsedResult['__exception']) {
            throw new Error(parsedResult['__exception']);
        }
        return parsedResult;
    }
    pushTask(param) {
        return new Promise((resolve, reject) => {
            this.taskQueue.push({ resolve, reject, param });
            this.runTaskLoop();
        });
    }
    async runTaskLoop() {
        if (this.busy) {
            return; // another loop is already running
        }
        this.busy = true;
        while (true) {
            const task = this.taskQueue.shift();
            if (!task)
                break; // no more tasks
            this.resultQueue.push(task);
            this.worker.postMessage(task.param);
        }
        this.busy = false;
    }
    onRecvMsg(e) {
        if (!e.data)
            return; // ignore
        const { verb, args } = e.data;
        if (verb && verb.startsWith('console.')) {
            if (verb.endsWith('log'))
                console.log(...args);
            if (verb.endsWith('warn'))
                console.warn(...args);
            if (verb.endsWith('error'))
                console.error(...args);
            return;
        }
        const { callbackId, result, err } = e.data;
        if (callbackId) {
            const idx = this.resultQueue.findIndex(t => t.param.callbackId === callbackId);
            if (idx !== -1) {
                const waitingTask = this.resultQueue.splice(idx, 1)[0];
                if (err)
                    waitingTask.reject(err);
                else
                    waitingTask.resolve(result);
            }
            else {
                console.error(`Cannot find waiting task with callbackId = ${callbackId}`);
            }
        }
    }
}
/**
 * Utility functions
 */
// Free ArrayBuffer by resizing them to 0. This is needed because sometimes we run into OOM issue.
function freeBuffer(buf) {
    // @ts-ignore
    if (ArrayBuffer.prototype.transfer) {
        // @ts-ignore
        buf.transfer(0);
        // @ts-ignore
    }
    else if (ArrayBuffer.prototype.resize && buf.resizable) {
        // @ts-ignore
        buf.resize(0);
    }
    else {
        console.warn('Cannot free buffer. You may run into out-of-memory issue.');
    }
}
// Zero-padding numbers
function padDigits(number, digits) {
    return Array(Math.max(digits - String(number).length + 1, 0)).join('0') + number;
}
//# sourceMappingURL=worker.js.map