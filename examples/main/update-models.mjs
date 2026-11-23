// resolves the models of models.json (same params as loadModelFromHF) and writes their URLs, sizes and capabilities
// to overlay/src/lib/wllama/models.generated.json, so the app does not call the HF API on each page load
// usage: node update-models.mjs (build wllama first, it uses esm/)

import { readFileSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { getHFModelSource } from '../../esm/index.js';
// same detection as the upstream models manager (Node runs these .ts files as is)
import { detectThinkingSupport } from '../../llama.cpp/ui/apps/chat/src/lib/utils/chat-template-thinking-detector.ts';
import { detectToolUseSupport } from '../../llama.cpp/ui/apps/chat/src/lib/utils/chat-template-tool-detector.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const INPUT = join(HERE, 'models.json');
const OUTPUT = join(HERE, 'overlay/src/lib/wllama/models.generated.json');
const HF = 'https://huggingface.co';
// mmproj quant picked when models.json does not give one
const MMPROJ_QUANTS = ['Q8_0', 'F16', 'BF16'];

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

async function fetchRange(url, size) {
  const res = await fetch(url, { headers: { Range: `bytes=0-${size - 1}` } });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

class NeedMoreData extends Error {}

// reads the key-value metadata of a GGUF header
function parseGgufKv(buf) {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = 0;
  const need = (n) => {
    if (pos + n > buf.byteLength) throw new NeedMoreData();
  };
  const u32 = () => (need(4), (pos += 4), view.getUint32(pos - 4, true));
  const u64 = () => (
    need(8), (pos += 8), Number(view.getBigUint64(pos - 8, true))
  );
  const str = () => {
    const n = u64();
    need(n);
    pos += n;
    return new TextDecoder().decode(buf.subarray(pos - n, pos));
  };
  const SIZES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
  };
  const value = (type) => {
    if (type === 7) return need(1), buf[pos++] !== 0;
    if (type === 4) return u32();
    if (type in SIZES) return need(SIZES[type]), (pos += SIZES[type]), null;
    if (type === 8) return str();
    if (type === 9) {
      const itemType = u32();
      const n = u64();
      for (let i = 0; i < n; i++) value(itemType);
      return null;
    }
    throw new Error(`unknown GGUF type ${type}`);
  };
  if (new TextDecoder().decode(buf.subarray(0, 4)) !== 'GGUF')
    throw new Error('not a GGUF file');
  pos = 4;
  u32(); // version
  u64(); // tensor count
  const nKv = u64();
  const kv = {};
  for (let i = 0; i < nKv; i++) {
    const key = str();
    kv[key] = value(u32());
  }
  return kv;
}

async function readGgufKv(url) {
  for (let size = 1 << 20; ; size *= 4) {
    try {
      return parseGgufKv(await fetchRange(url, size));
    } catch (e) {
      if (!(e instanceof NeedMoreData) || size > 64 << 20) throw e;
    }
  }
}

// size of a file, or of all the shards of a split model
function totalSize(files, path) {
  const shard = path.match(/^(.*)-\d{5}-of-(\d{5})\.gguf$/);
  const paths = shard
    ? Array.from(
        { length: Number(shard[2]) },
        (_, i) =>
          `${shard[1]}-${String(i + 1).padStart(5, '0')}-of-${shard[2]}.gguf`
      )
    : [path];
  return paths.reduce((sum, p) => {
    const f = files.find((f) => f.path === p);
    if (!f) throw new Error(`file not found: ${p}`);
    return sum + (f.lfs?.size ?? f.size);
  }, 0);
}

async function resolve(params) {
  const { repo } = params;
  const files = await getJson(
    `${HF}/api/models/${repo}/tree/main?recursive=true`
  );
  const mmprojs = files
    .filter((f) => /mmproj.*\.gguf$/i.test(f.path))
    .map((f) => f.path);
  const hfParams = { ...params };
  if (
    params.mmprojQuant === undefined &&
    params.mmprojFile === undefined &&
    mmprojs.length > 0
  ) {
    // whole token match: "F16" must not match "BF16"
    const hasQuant = (p, q) =>
      new RegExp(`(^|[-_./])${q}([-_.]|$)`, 'i').test(p);
    const quant = MMPROJ_QUANTS.find((q) =>
      mmprojs.some((p) => hasQuant(p, q))
    );
    hfParams.mmprojFile =
      mmprojs.find((p) => quant && hasQuant(p, quant)) ?? mmprojs[0];
  }
  // same file selection as loadModelFromHF
  const source = await getHFModelSource(hfParams);
  const pathOf = (url) => url.slice(`${HF}/${repo}/resolve/main/`.length);
  let size = totalSize(files, pathOf(source.url));

  const modalities = [];
  if (source.mmprojUrl) {
    size += totalSize(files, pathOf(source.mmprojUrl));
    const kv = await readGgufKv(source.mmprojUrl);
    if (kv['clip.has_vision_encoder']) modalities.push('image');
    if (kv['clip.has_audio_encoder']) modalities.push('audio');
  }
  const { gguf } = await getJson(`${HF}/api/models/${repo}?expand[]=gguf`);
  const template = gguf?.chat_template ?? '';

  return {
    repo,
    url: source.url,
    ...(source.mmprojUrl ? { mmprojUrl: source.mmprojUrl } : {}),
    size,
    modalities,
    tools: detectToolUseSupport(template),
    thinking: detectThinkingSupport(template),
  };
}

const input = JSON.parse(readFileSync(INPUT, 'utf8'));
const models = [];
for (const params of input) {
  const m = await resolve(params);
  const caps = [
    ...m.modalities,
    ...(m.tools ? ['tools'] : []),
    ...(m.thinking ? ['thinking'] : []),
  ];
  console.log(
    `${m.url.split('/').pop()}  ${(m.size / 1e9).toFixed(2)} GB  [${caps.join(', ')}]`
  );
  models.push(m);
}
models.sort((a, b) => a.size - b.size);
writeFileSync(OUTPUT, JSON.stringify(models, null, 2) + '\n');
console.log(`wrote ${models.length} models to ${OUTPUT}`);
