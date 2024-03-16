import ModuleSingleThread from './single-thread/wllama';
import ModuleMultiThread from './multi-thread/wllama';
import { AssetsPathConfig } from './wllama';

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
      msg({ verb: 'console.log', args: [\`Loading "\${filename}" from "\${p}"\`] });
      return p;
    },
  };
};


// Start the main llama.cpp
let wModule;
let wllamaStart;
let wllamaAction;
let wllamaExit;

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
    const argGGUFBuffer = args[0]; // buffer for model
    try {
      const Module = ModuleWrapper();
      wModule = await Module(getWModuleConfig(pathConfig));
      // init FS
      wModule['FS_createPath']('/', 'models', true, true);
      wModule['FS_createDataFile']('/models', 'model.bin', argGGUFBuffer, true, true, true);
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
  worker: Worker;

  constructor(pathConfig: any, multiThread: boolean = false) {
    let moduleCode = multiThread ? ModuleMultiThread.toString() : ModuleSingleThread.toString();
    // monkey-patch: remove all "import.meta"
    // FIXME: this monkey-patch will remove support for nodejs
    moduleCode = moduleCode.replace(/import\.meta/g, 'importMeta');
    const completeCode = [
      'const importMeta = {}',
      `const pathConfig = ${JSON.stringify(pathConfig)}`,
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
  }

  moduleInit(ggufBuffer: Uint8Array): Promise<void> {
    return this.pushTask({
      verb: 'module.init',
      args: [ggufBuffer],
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
      this.worker.postMessage(task.param);
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
