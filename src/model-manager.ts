import CacheManager, {
  type CacheEntry,
  type DownloadOptions,
} from './cache-manager';
import { isValidGgufFile, sumArr } from './utils';
import { WllamaError, type WllamaLogger } from './wllama';

const DEFAULT_PARALLEL_DOWNLOADS = 3;

/**
 * Callback function to track download progress
 */
export type DownloadProgressCallback = (opts: {
  /**
   * Number of bytes loaded (sum of all shards)
   */
  loaded: number;
  /**
   * Total number of bytes (sum of all shards)
   */
  total: number;
}) => any;

/**
 * Status of the model validation
 */
export enum ModelValidationStatus {
  VALID = 'valid',
  INVALID = 'invalid',
  DELETED = 'deleted',
}

/**
 * Parameters for ModelManager constructor
 */
export interface ModelManagerParams {
  cacheManager?: CacheManager;
  logger?: WllamaLogger;
  /**
   * Number of parallel downloads
   *
   * Default: 3
   */
  parallelDownloads?: number | undefined;
  /**
   * Allow offline mode
   *
   * Default: false
   */
  allowOffline?: boolean | undefined;
}

/**
 * Model class
 *
 * One model can have multiple shards, each shard is a GGUF file.
 */
export class Model {
  private modelManager: ModelManager;
  constructor(
    modelManager: ModelManager,
    url: string,
    savedFiles?: CacheEntry[]
  ) {
    this.modelManager = modelManager;
    this.url = url;
    if (savedFiles) {
      // this file is already in cache
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      // this file is not in cache, we are about to download it
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
   *
   * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
   */
  url: string;
  /**
   * Size in bytes (total size of all shards).
   *
   * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
   */
  size: number;
  /**
   * List of all shards in the cache, sorted by original URL (ascending order)
   */
  files: CacheEntry[];
  /**
   * Open and get a list of all shards as Blobs
   */
  async open(): Promise<Blob[]> {
    if (this.size === -1) {
      throw new WllamaError(
        `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
        'load_error'
      );
    }
    const blobs: Blob[] = [];
    for (const file of this.files) {
      const blob = await this.modelManager.cacheManager.open(file.name);
      if (!blob) {
        throw new Error(
          `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
        );
      }
      blobs.push(blob);
    }
    return blobs;
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate(): ModelValidationStatus {
    const nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.size === -1) {
      return ModelValidationStatus.DELETED;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return ModelValidationStatus.INVALID;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return ModelValidationStatus.INVALID;
      }
    }
    return ModelValidationStatus.VALID;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  async refresh(options: DownloadOptions = {}): Promise<void> {
    const urls = ModelManager.parseModelUrl(this.url);
    const works = urls.map((url, index) => ({
      url,
      index,
    }));
    this.modelManager.logger.debug('Downloading model files:', urls);
    const nParallel =
      this.modelManager.params.parallelDownloads ?? DEFAULT_PARALLEL_DOWNLOADS;
    const totalSize = await this.getTotalDownloadSize(urls);
    const loadedSize: number[] = [];
    const worker = async () => {
      while (works.length > 0) {
        const w = works.shift();
        if (!w) break;
        await this.modelManager.cacheManager.download(w.url, {
          ...options,
          progressCallback: ({ loaded }) => {
            loadedSize[w.index] = loaded;
            options.progressCallback?.({
              loaded: sumArr(loadedSize),
              total: totalSize,
            });
          },
        });
      }
    };
    const promises: Promise<void>[] = [];
    for (let i = 0; i < nParallel; i++) {
      promises.push(worker());
      loadedSize.push(0);
    }
    await Promise.all(promises);
    this.files = this.getAllFiles(await this.modelManager.cacheManager.list());
    this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
  }
  /**
   * Remove the model from the cache
   */
  async remove(): Promise<void> {
    this.files = this.getAllFiles(await this.modelManager.cacheManager.list());
    await this.modelManager.cacheManager.deleteMany(
      (f) => !!this.files.find((file) => file.name === f.name)
    );
    this.size = -1;
  }

  private getAllFiles(savedFiles: CacheEntry[]): CacheEntry[] {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    const allFiles: CacheEntry[] = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort((a, b) =>
      a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }

  private async getTotalDownloadSize(urls: string[]): Promise<number> {
    const responses = await Promise.all(
      urls.map((url) => fetch(url, { method: 'HEAD' }))
    );
    const sizes = responses.map((res) =>
      Number(res.headers.get('content-length') || '0')
    );
    return sumArr(sizes);
  }
}

export class ModelManager {
  // The CacheManager singleton, can be accessed by user
  public cacheManager: CacheManager;

  public params: ModelManagerParams;
  public logger: WllamaLogger;

  constructor(params: ModelManagerParams = {}) {
    this.cacheManager = params.cacheManager || new CacheManager();
    this.params = params;
    this.logger = params.logger || console;
  }

  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl: string | string[]): string[] {
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = queryMatch?.[1] ?? '';
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, '');
    const total = matches[2];
    const paddedShardIds = Array.from({ length: Number(total) }, (_, index) =>
      (index + 1).toString().padStart(5, '0')
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }

  /**
   * Get all models in the cache
   */
  async getModels(opts: { includeInvalid?: boolean } = {}): Promise<Model[]> {
    const cachedFiles = await this.cacheManager.list();
    let models: Model[] = [];
    for (const file of cachedFiles) {
      const shards = ModelManager.parseModelUrl(file.metadata.originalURL);
      const isFirstShard =
        shards.length === 1 || shards[0] === file.metadata.originalURL;
      if (isFirstShard) {
        models.push(new Model(this, file.metadata.originalURL, cachedFiles));
      }
    }
    if (!opts.includeInvalid) {
      models = models.filter(
        (m) => m.validate() === ModelValidationStatus.VALID
      );
    }
    return models;
  }

  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  async downloadModel(
    url: string,
    options: DownloadOptions = {}
  ): Promise<Model> {
    if (!isValidGgufFile(url)) {
      throw new WllamaError(
        `Invalid model URL: ${url}; URL must ends with ".gguf"`,
        'download_error'
      );
    }
    const model = new Model(this, url, undefined);
    const validity = model.validate();
    if (validity !== ModelValidationStatus.VALID) {
      await model.refresh(options);
    }
    return model;
  }

  /**
   * Get a model from the cache or download it if it's not available.
   */
  async getModelOrDownload(
    url: string,
    options: DownloadOptions = {}
  ): Promise<Model> {
    const models = await this.getModels();
    const model = models.find((m) => m.url === url);
    if (model) {
      options.progressCallback?.({ loaded: model.size, total: model.size });
      return model;
    }
    return this.downloadModel(url, options);
  }

  /**
   * Remove all models from the cache
   */
  async clear(): Promise<void> {
    await this.cacheManager.clear();
  }
}
