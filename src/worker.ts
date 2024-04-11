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
const getWModuleConfig = (pathConfig) => {
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
  };
};


// Start the main llama.cpp
let wModule;
let wllamaStart;
let wllamaAction;
let wllamaExit;

// utility function
function padDigits(number, digits) {
  return Array(Math.max(digits - String(number).length + 1, 0)).join(0) + number;
}

const callWrapper = (name, ret, args) => {
  const fn = wModule.cwrap(name, ret, args);
  const decodeException = wModule.cwrap('wllama_decode_exception', 'string', ['number']);
  return async (action, req) => {
    let result;
    try {
      if (args.length === 2) {
        result = await fn(action, req);
      } else {
        result = fn();
      }
    } catch (ex) {
      const what = await decodeException(ex);
      console.error(what);
      throw new Error(what);
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
    const argGGUFBuffers = args[0]; // buffers for model
    try {
      const Module = ModuleWrapper();
      wModule = await Module(getWModuleConfig(pathConfig));

      // init FS
      wModule['FS_createPath']('/', 'models', true, true);
      if (argGGUFBuffers.length === 1) {
        wModule['FS_createDataFile']('/models', 'model.gguf', argGGUFBuffers[0], true, true, true);
      } else {
        for (let i = 0; i < argGGUFBuffers.length; i++) {
          const fname = 'model-' + padDigits(i + 1, 5) + '-of-' + padDigits(argGGUFBuffers.length, 5) + '.gguf';
          wModule['FS_createDataFile']('/models', fname, argGGUFBuffers[i], true, true, true);
        }
      }

      // init cwrap
      wllamaStart  = callWrapper('wllama_start' , 'number', []);
      wllamaAction = callWrapper('wllama_action', 'string', ['string', 'string']);
      wllamaExit   = callWrapper('wllama_exit'  , 'number', []);
      msg({ callbackId, result: null });

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

interface TaskParam {
  verb: 'module.init' | 'wllama.start' | 'wllama.action' | 'wllama.exit',
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

  constructor(pathConfig: any, multiThread: boolean = false) {
    this.pathConfig = pathConfig;
    this.multiThread = multiThread;
  }

  async moduleInit(ggufBuffers: Uint8Array[]): Promise<void> {
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
      `const pathConfig = ${JSON.stringify(this.pathConfig)}`,
      `function ModuleWrapper() {
        const _scriptDir = ${JSON.stringify(window.location.href)};
        return ${moduleCode};
      }`,
      WORKER_CODE,
    ].join(';\n\n');
    // https://stackoverflow.com/questions/5408406/web-workers-without-a-separate-javascript-file
    const workerURL = window.URL.createObjectURL(new Blob([completeCode], {type: 'text/javascript'}));
    this.worker = new Worker(workerURL);
    this.worker.onmessage = this.onRecvMsg.bind(this);
    this.worker.onerror = console.error;

    return await this.pushTask({
      verb: 'module.init',
      args: [ggufBuffers],
      callbackId: this.taskId++,
    });
  }

  wllamaStart(): Promise<number> {
    return this.pushTask({
      verb: 'wllama.start',
      args: [],
      callbackId: this.taskId++,
    });
  }

  async wllamaAction(name: string, body: any): Promise<any> {
    const result = await this.pushTask({
      verb: 'wllama.action',
      args: [name, JSON.stringify(body)],
      callbackId: this.taskId++,
    });
    return JSON.parse(result);
  }

  wllamaExit(): Promise<number> {
    return this.pushTask({
      verb: 'wllama.exit',
      args: [],
      callbackId: this.taskId++,
    });
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
}
