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
interface TaskParam {
    verb: 'module.init' | 'wllama.start' | 'wllama.action' | 'wllama.exit';
    args: any[];
    callbackId: number;
}
interface Task {
    resolve: any;
    reject: any;
    param: TaskParam;
}
export declare class ProxyToWorker {
    taskQueue: Task[];
    taskId: number;
    resultQueue: Task[];
    busy: boolean;
    worker?: Worker;
    pathConfig: any;
    multiThread: boolean;
    constructor(pathConfig: any, multiThread?: boolean);
    moduleInit(ggufBuffers: Uint8Array[]): Promise<void>;
    wllamaStart(): Promise<number>;
    wllamaAction(name: string, body: any): Promise<any>;
    wllamaExit(): Promise<number>;
    private pushTask;
    private runTaskLoop;
    private onRecvMsg;
}
export {};
