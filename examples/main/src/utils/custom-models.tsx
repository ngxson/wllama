import { MAX_GGUF_SIZE } from '../config';
import { DisplayedModel } from './displayed-model';
import { WllamaStorage } from './utils';

const ggufMagicNumber = new Uint8Array([0x47, 0x47, 0x55, 0x46]);

export async function verifyCustomModel(url: string): Promise<DisplayedModel> {
  const _url = url.replace(/\?.*/, '');

  const response = await fetch(_url, {
    headers: {
      Range: `bytes=0-${2 * 1024 * 1024}`,
    },
  });

  if (response.ok) {
    const buf = await response.arrayBuffer();
    if (!checkBuffer(new Uint8Array(buf.slice(0, 4)), ggufMagicNumber)) {
      throw new Error(
        'Not a valid gguf file: not starting with GGUF magic number'
      );
    }
  } else {
    throw new Error(`Fetch error with status code = ${response.status}`);
  }

  return new DisplayedModel(_url, await getModelSize(_url), true, undefined);
}

const checkBuffer = (buffer: Uint8Array, header: Uint8Array) => {
  for (let i = 0; i < header.length; i++) {
    if (header[i] !== buffer[i]) {
      return false;
    }
  }
  return true;
};

const getModelSize = async (url: string): Promise<number> => {
  const urls = parseModelUrl(url);

  const sizes = await Promise.all(
    urls.map(async (url) => {
      const response = await fetch(url, {
        method: 'HEAD',
      });

      if (response.ok) {
        const contentLength = response.headers.get('Content-Length');
        if (contentLength) {
          return parseInt(contentLength);
        } else {
          return 0;
        }
      } else {
        throw new Error(`Fetch error with status code = ${response.status}`);
      }
    })
  );

  if (sizes.some((s) => s >= MAX_GGUF_SIZE)) {
    throw new Error(
      'GGUF file is too big (max. 2GB per file). Please split the file into smaller shards (learn more in "Guide")'
    );
  }

  return sumArr(sizes);
};

const parseModelUrl = (modelUrl: string): string[] => {
  const urlPartsRegex =
    /(?<baseURL>.*)-(?<current>\d{5})-of-(?<total>\d{5})\.gguf$/;
  const matches = modelUrl.match(urlPartsRegex);
  if (!matches || !matches.groups || Object.keys(matches.groups).length !== 3) {
    return [modelUrl];
  }
  const { baseURL, total } = matches.groups;
  const paddedShardIds = Array.from({ length: Number(total) }, (_, index) =>
    (index + 1).toString().padStart(5, '0')
  );
  return paddedShardIds.map(
    (current) => `${baseURL}-${current}-of-${total}.gguf`
  );
};

const sumArr = (arr: number[]) => arr.reduce((sum, num) => sum + num, 0);

// for debugging only
// @ts-ignore
window._exportModelList = function () {
  const list: any[] = WllamaStorage.load('custom_models', []);
  const listExported = list.map((m) => {
    delete m.userAdded;
    return m;
  });
  console.log(JSON.stringify(listExported, null, 2));
};
