import CacheManager, { CacheEntry, DownloadOptions } from './cache-manager';
import { sumArr } from './utils';
import { WllamaError, WllamaLogger } from './wllama';

const DEFAULT_PARALLEL_DOWNLOADS = 3;

export type DownloadProgressCallback = (opts: {
  loaded: number;
  total: number;
}) => any;

export enum ModelValidationStatus {
  VALID,
  INVALID,
  DELETED,
}

export interface ModelManagerParams {
  cacheManager?: CacheManager;
  logger?: WllamaLogger;
  parallelDownloads?: number;
  allowOffline?: boolean;
}

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
   * This of all GGUF files
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
  async validate(): Promise<ModelValidationStatus> {
    // TODO: valid number of shards
    if (this.size === -1) {
      return ModelValidationStatus.DELETED;
    }
    if (this.size < 16) {
      return ModelValidationStatus.INVALID;
    }
    for (const file of this.files) {
      const metadata = await this.modelManager.cacheManager.getMetadata(
        file.name
      );
      if (!metadata || metadata.originalSize !== file.size) {
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
    await this.modelManager.cacheManager.deleteMany((f) =>
      this.files.includes(f)
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

  constructor(params: ModelManagerParams) {
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
    const urlPartsRegex =
      /(?<baseURL>.*)-(?<current>\d{5})-of-(?<total>\d{5})\.gguf$/;
    const matches = modelUrl.match(urlPartsRegex);
    if (
      !matches ||
      !matches.groups ||
      Object.keys(matches.groups).length !== 3
    ) {
      return [modelUrl];
    }
    const { baseURL, total } = matches.groups;
    const paddedShardIds = Array.from({ length: Number(total) }, (_, index) =>
      (index + 1).toString().padStart(5, '0')
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf`
    );
  }

  /**
   * Get all models in the cache
   */
  async getModels(): Promise<Model[]> {
    const cachedFiles = await this.cacheManager.list();
    const models: Model[] = [];
    for (const file of cachedFiles) {
      const shards = ModelManager.parseModelUrl(file.metadata.originalURL);
      const isFirstShard =
        shards.length === 1 || shards[0] === file.metadata.originalURL;
      if (isFirstShard) {
        models.push(new Model(this, file.metadata.originalURL, cachedFiles));
      }
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
    if (!url.endsWith('.gguf')) {
      throw new WllamaError(
        `Invalid model URL: ${url}; URL must ends with ".gguf"`,
        'download_error'
      );
    }
    const model = new Model(this, url, undefined);
    const validity = await model.validate();
    if (validity !== ModelValidationStatus.VALID) {
      await model.refresh(options);
    }
    return model;
  }

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
}
