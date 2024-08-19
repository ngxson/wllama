import CacheManager from '../cache-manager';
import { GGUFRemoteBlob } from './remote-blob';

type ProgressCallback = (opts: { loaded: number; total: number }) => any;
interface Task {
  url: string;
  state: State;
  signalStart: Promise<void>;
  fireStart(): void;
  signalEnd: Promise<void>;
  fireEnd(): void;
  blob: Blob;
  loaded: number;
}
enum State {
  READY,
  WORKING,
  FINISHED,
}

export class MultiDownloads {
  private tasks: Task[];
  private maxParallel: number;
  private progressCallback?: ProgressCallback;
  private logger: any;
  private useCache: boolean;
  private totalBytes: number = 0;
  private allowOffline: boolean;
  private noTEE: boolean;
  private cacheManager: CacheManager;

  constructor(
    logger: any,
    urls: string[],
    maxParallel: number,
    cacheManager: CacheManager,
    opts: {
      progressCallback?: ProgressCallback;
      useCache: boolean;
      allowOffline: boolean;
      noTEE?: boolean;
    }
  ) {
    this.tasks = urls.map((url) => {
      // @ts-ignore
      const task: Task = {
        url,
        state: State.READY,
        loaded: 0,
      };
      task.signalStart = new Promise((resolve) => (task.fireStart = resolve));
      task.signalEnd = new Promise((resolve) => (task.fireEnd = resolve));
      return task;
    });
    this.logger = logger;
    this.maxParallel = maxParallel;
    this.progressCallback = opts.progressCallback;
    this.useCache = opts.useCache;
    this.allowOffline = opts.allowOffline;
    this.noTEE = !!opts.noTEE;
    this.cacheManager = cacheManager;
  }

  async run(): Promise<Blob[]> {
    // create all Blobs
    await Promise.all(
      this.tasks.map(async (task) => {
        task.blob = await GGUFRemoteBlob.create(task.url, {
          logger: this.logger,
          useCache: this.useCache,
          startSignal: task.signalStart,
          allowOffline: this.allowOffline,
          noTEE: this.noTEE,
          cacheManager: this.cacheManager,
          progressCallback: ({ loaded }) => {
            task.loaded = loaded;
            this.updateProgress(task);
          },
        });
      })
    );
    // calculate totalBytes
    this.totalBytes = this.tasks.reduce((n, task) => n + task.blob.size, 0);
    // run N dispatchers
    for (let i = 0; i < this.maxParallel; i++) {
      this.dispatcher();
    }
    return this.tasks.map((t) => t.blob);
  }

  updateProgress(task: Task) {
    const progress = {
      loaded: this.tasks.reduce((n, task) => n + task.loaded, 0),
      total: this.totalBytes,
    };
    this.progressCallback?.(progress);
    if (task.loaded === task.blob.size) {
      // task finished
      task.state = State.FINISHED;
      task.fireEnd();
    }
  }

  async dispatcher() {
    while (true) {
      const task = this.tasks.find((t) => t.state === State.READY);
      if (!task) return;
      task.state = State.WORKING;
      task.fireStart();
      await task.signalEnd;
    }
  }
}
