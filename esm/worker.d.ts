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
    worker: Worker;
    constructor(pathConfig: any, multiThread?: boolean);
    moduleInit(ggufBuffer: Uint8Array): Promise<void>;
    wllamaStart(): Promise<number>;
    wllamaAction(name: string, body: any): Promise<any>;
    wllamaExit(): Promise<number>;
    private pushTask;
    private runTaskLoop;
    private onRecvMsg;
}
export {};
