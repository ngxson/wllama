import { Model } from '@wllama/wllama';
import { ModelState } from './types';
import { WllamaStorage } from './utils';
import { LIST_MODELS } from '../config';

export class DisplayedModel {
  url: string;
  size: number;
  isUserAdded: boolean;
  cachedModel?: Model;

  state: ModelState = ModelState.NOT_DOWNLOADED;
  downloadPercent: number = -1; // from 0.0 to 1.0; -1 means not downloading

  constructor(
    url: string,
    size: number,
    isUserAdded: boolean,
    cachedModel?: Model
  ) {
    this.url = url;
    this.size = size;
    this.isUserAdded = isUserAdded;
    this.state = !!cachedModel ? ModelState.READY : ModelState.NOT_DOWNLOADED;
    this.cachedModel = cachedModel;
  }

  get hfModel() {
    const parts = this.url
      .replace(/https:\/\/(huggingface.co|hf.co)\/+/, '')
      .split('/');
    return `${parts[0]}/${parts[1]}`;
  }

  get hfPath() {
    const parts = this.url
      .replace(/https:\/\/(huggingface.co|hf.co)\/+/, '')
      .split('/');
    return parts.slice(4).join('/');
  }

  clone(overwrite: Partial<DisplayedModel>): DisplayedModel {
    const obj = new DisplayedModel(
      this.url,
      this.size,
      this.isUserAdded,
      this.cachedModel
    );
    obj.state = overwrite.state ?? this.state;
    obj.downloadPercent = overwrite.downloadPercent ?? this.downloadPercent;
    return obj;
  }
}

interface UserAddedModel {
  url: string;
  size: number;
}

export function getUserAddedModels(cachedModels: Model[]): DisplayedModel[] {
  const userAddedModels: UserAddedModel[] = WllamaStorage.load(
    'custom_models',
    []
  );
  return userAddedModels.map((m: any) => {
    const cachedModel = cachedModels.find((cm) => cm.url === m.url);
    return new DisplayedModel(m.url, m.size, true, cachedModel);
  });
}

export function updateUserAddedModels(models: DisplayedModel[]) {
  const userAddedModels: UserAddedModel[] = models
    .filter((m) => m.isUserAdded)
    .map((m) => ({ url: m.url, size: m.size }));
  WllamaStorage.save('custom_models', userAddedModels);
}

export function getPresetModels(cachedModels: Model[]): DisplayedModel[] {
  return LIST_MODELS.map((m) => {
    const cachedModel = cachedModels.find((cm) => cm.url === m.url);
    return new DisplayedModel(m.url, m.size, false, cachedModel);
  });
}

export function getDisplayedModels(cachedModels: Model[]): DisplayedModel[] {
  return [
    ...getUserAddedModels(cachedModels),
    ...getPresetModels(cachedModels),
  ];
}
