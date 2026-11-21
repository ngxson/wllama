// The wasm uses 32-bit memory, so llama.cpp cannot read a file of 4GB or more (file offsets are size_t).
// A bigger GGUF is split into virtual shards, the same as llama-gguf-split does: each shard is a new header plus a slice of the original file, no data is copied.

export const GGUF_SPLIT_CONFIG = {
  // files of this size or bigger are split
  maxFileSize: 4 * 1024 * 1024 * 1024 - 1,
  // target data size of one shard
  shardSize: 2 * 1024 * 1024 * 1024,
};

const GGUF_MAGIC = 0x46554747; // "GGUF"
const KEY_ALIGNMENT = 'general.alignment';
const KEY_SPLIT_NO = 'split.no';
const KEY_SPLIT_COUNT = 'split.count';
const KEY_SPLIT_TENSORS_COUNT = 'split.tensors.count';
const KEY_TOKENS = 'tokenizer.ggml.tokens';

enum GgufType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

const TYPE_SIZE: Record<number, number> = {
  [GgufType.UINT8]: 1,
  [GgufType.INT8]: 1,
  [GgufType.UINT16]: 2,
  [GgufType.INT16]: 2,
  [GgufType.UINT32]: 4,
  [GgufType.INT32]: 4,
  [GgufType.FLOAT32]: 4,
  [GgufType.BOOL]: 1,
  [GgufType.UINT64]: 8,
  [GgufType.INT64]: 8,
  [GgufType.FLOAT64]: 8,
};

interface GgufKv {
  key: string;
  raw: Uint8Array; // the whole KV: key, type and value
}

interface GgufTensorInfo {
  name: string;
  raw: Uint8Array; // name, n_dims, dims and type, without the offset
  offset: number; // relative to the data section
}

interface GgufHeader {
  version: number;
  splitCount: number; // 0 if the key is missing
  nVocab: number; // 0 if the key is missing
  kvs: GgufKv[];
  tensors: GgufTensorInfo[];
  alignment: number;
  dataStart: number;
}

// thrown when the buffer is too small, the caller reads more of the file and tries again
class NeedMoreData extends Error {}

class Reader {
  pos = 0;
  private view: DataView;
  constructor(private bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  private need(n: number) {
    if (this.pos + n > this.bytes.byteLength) throw new NeedMoreData();
  }
  u32() {
    this.need(4);
    const v = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return v;
  }
  u16() {
    this.need(2);
    const v = this.view.getUint16(this.pos, true);
    this.pos += 2;
    return v;
  }
  u64() {
    this.need(8);
    const v = Number(this.view.getBigUint64(this.pos, true));
    this.pos += 8;
    return v;
  }
  skip(n: number) {
    this.need(n);
    this.pos += n;
  }
  str() {
    const len = this.u64();
    this.need(len);
    const s = new TextDecoder().decode(
      this.bytes.subarray(this.pos, this.pos + len)
    );
    this.pos += len;
    return s;
  }
  skipValue(type: number) {
    if (type === GgufType.STRING) {
      this.skip(this.u64());
    } else if (type === GgufType.ARRAY) {
      const itemType = this.u32();
      const n = this.u64();
      if (itemType === GgufType.STRING || itemType === GgufType.ARRAY) {
        for (let i = 0; i < n; i++) this.skipValue(itemType);
      } else {
        this.skip(n * this.typeSize(itemType));
      }
    } else {
      this.skip(this.typeSize(type));
    }
  }
  slice(start: number) {
    return this.bytes.slice(start, this.pos);
  }
  private typeSize(type: number) {
    const size = TYPE_SIZE[type];
    if (size === undefined) throw new Error(`Unknown GGUF type ${type}`);
    return size;
  }
}

const parseHeader = (bytes: Uint8Array): GgufHeader => {
  const r = new Reader(bytes);
  if (r.u32() !== GGUF_MAGIC) throw new Error('Not a GGUF file');
  const version = r.u32();
  if (version < 2) throw new Error(`Unsupported GGUF version ${version}`);
  const nTensors = r.u64();
  const nKv = r.u64();
  const kvs: GgufKv[] = [];
  let alignment = 32;
  let splitCount = 0;
  let nVocab = 0;
  for (let i = 0; i < nKv; i++) {
    const start = r.pos;
    const key = r.str();
    const type = r.u32();
    if (key === KEY_ALIGNMENT && type === GgufType.UINT32) {
      alignment = r.u32();
    } else if (key === KEY_SPLIT_COUNT && type === GgufType.UINT16) {
      splitCount = r.u16();
    } else if (key === KEY_TOKENS && type === GgufType.ARRAY) {
      const pos = r.pos;
      r.u32(); // item type
      nVocab = r.u64();
      r.pos = pos;
      r.skipValue(type);
    } else {
      r.skipValue(type);
    }
    kvs.push({ key, raw: r.slice(start) });
  }
  const tensors: GgufTensorInfo[] = [];
  for (let i = 0; i < nTensors; i++) {
    const start = r.pos;
    const name = r.str();
    const nDims = r.u32();
    r.skip(nDims * 8 + 4); // dims and type
    const raw = r.slice(start);
    tensors.push({ name, raw, offset: r.u64() });
  }
  const dataStart = Math.ceil(r.pos / alignment) * alignment;
  return { version, splitCount, nVocab, kvs, tensors, alignment, dataStart };
};

const readHeader = async (blob: Blob): Promise<GgufHeader> => {
  let size = 8 * 1024 * 1024;
  while (true) {
    size = Math.min(size, blob.size);
    const bytes = new Uint8Array(await blob.slice(0, size).arrayBuffer());
    try {
      return parseHeader(bytes);
    } catch (e) {
      if (!(e instanceof NeedMoreData) || size === blob.size) throw e;
      size *= 4;
    }
  }
};

class Writer {
  private parts: Uint8Array[] = [];
  private length = 0;
  bytes(b: Uint8Array) {
    this.parts.push(b);
    this.length += b.byteLength;
  }
  private num(n: number, size: number, write: (v: DataView) => void) {
    const b = new Uint8Array(size);
    write(new DataView(b.buffer));
    this.bytes(b);
  }
  u16(v: number) {
    this.num(v, 2, (d) => d.setUint16(0, v, true));
  }
  u32(v: number) {
    this.num(v, 4, (d) => d.setUint32(0, v, true));
  }
  i32(v: number) {
    this.num(v, 4, (d) => d.setInt32(0, v, true));
  }
  u64(v: number) {
    this.num(v, 8, (d) => d.setBigUint64(0, BigInt(v), true));
  }
  str(s: string) {
    const b = new TextEncoder().encode(s);
    this.u64(b.byteLength);
    this.bytes(b);
  }
  // header padded with zeros to the alignment
  finish(alignment: number): Uint8Array {
    const padded = Math.ceil(this.length / alignment) * alignment;
    const out = new Uint8Array(padded);
    let pos = 0;
    for (const p of this.parts) {
      out.set(p, pos);
      pos += p.byteLength;
    }
    return out;
  }
}

/**
 * Split a GGUF file of `GGUF_SPLIT_CONFIG.maxFileSize` or more into virtual shards named `<name>-0000k-of-0000n.gguf`.
 * Smaller files and files that are already a split are returned as they are.
 */
export const splitBigGguf = async (blob: Blob): Promise<Blob[]> => {
  const { maxFileSize, shardSize } = GGUF_SPLIT_CONFIG;
  if (blob.size < maxFileSize) return [blob];
  const header = await readHeader(blob);
  // a merged file can keep split.count = 1, its split keys are dropped below
  if (header.splitCount > 1) {
    throw new Error(
      'A shard of a split GGUF is bigger than 4GB, please split the model into smaller shards'
    );
  }
  const { tensors, alignment, dataStart } = header;
  const dataSize = blob.size - dataStart;

  // tensors are stored in the order of the header, each one padded to the alignment
  const groups: GgufTensorInfo[][] = [];
  let groupStart = 0;
  tensors.forEach((t, i) => {
    const end = i + 1 < tensors.length ? tensors[i + 1].offset : dataSize;
    if (end < t.offset || (i === 0 && t.offset !== 0)) {
      throw new Error('GGUF tensors are not stored in order');
    }
    if (groups.length === 0 || end - groupStart > shardSize) {
      groups.push([]);
      groupStart = t.offset;
    }
    groups[groups.length - 1].push(t);
  });

  const alignmentKv = header.kvs.find((kv) => kv.key === KEY_ALIGNMENT);
  const baseName = ((blob as File).name || 'model').replace(/\.gguf$/i, '');
  const pad = (n: number) => String(n).padStart(5, '0');
  return groups.map((group, k) => {
    const start = group[0].offset;
    const end = k + 1 < groups.length ? groups[k + 1][0].offset : dataSize;
    // same as llama-gguf-split: all metadata in the first shard, only the split keys in the others
    const kvs =
      k === 0
        ? header.kvs.filter((kv) => !kv.key.startsWith('split.'))
        : alignmentKv
          ? [alignmentKv]
          : [];
    const w = new Writer();
    w.u32(GGUF_MAGIC);
    w.u32(header.version);
    w.u64(group.length);
    w.u64(kvs.length + 3);
    for (const kv of kvs) w.bytes(kv.raw);
    w.str(KEY_SPLIT_NO);
    w.u32(GgufType.UINT16);
    w.u16(k);
    w.str(KEY_SPLIT_COUNT);
    w.u32(GgufType.UINT16);
    w.u16(groups.length);
    w.str(KEY_SPLIT_TENSORS_COUNT);
    w.u32(GgufType.INT32);
    w.i32(tensors.length);
    for (const t of group) {
      w.bytes(t.raw);
      w.u64(t.offset - start);
    }
    const shardHeader = w.finish(alignment);
    const data = blob.slice(dataStart + start, dataStart + end);
    return new File(
      [shardHeader, data],
      `${baseName}-${pad(k + 1)}-of-${pad(groups.length)}.gguf`
    );
  });
};

/**
 * Size in bytes of each tensor of a GGUF file (with its padding), and the vocab size (0 if the file has no tokenizer)
 */
export const readGgufTensorSizes = async (
  blob: Blob
): Promise<{ nVocab: number; tensors: { name: string; size: number }[] }> => {
  const { nVocab, tensors, dataStart } = await readHeader(blob);
  const dataSize = blob.size - dataStart;
  return {
    nVocab,
    tensors: tensors.map((t, i) => ({
      name: t.name,
      size:
        (i + 1 < tensors.length ? tensors[i + 1].offset : dataSize) - t.offset,
    })),
  };
};

// a wasm instance has 4GB of memory, the budgets leave room for the KV cache and the compute buffers
export const RPC_BUDGET = {
  single: 3.5 * 1024 * 1024 * 1024,
  worker: 2.5 * 1024 * 1024 * 1024,
};
const MAX_RPC_WORKERS = 16; // GGML_RPC_MAX_SERVERS

/**
 * Number of rpc workers needed so that each wasm instance fits in its memory, 0 if one instance is enough.
 * The main instance keeps the input embeddings, the rpc workers get the layers and the output.
 * token_embd goes with the layers, same as with token_embd_on_gpu (the default).
 */
export const countRpcWorkers = (
  tensors: { name: string; size: number }[],
  nVocab: number,
  nUbatch: number
): number => {
  const isInput = (name: string) =>
    /^(per_layer_token_embd|position_embd|token_types)\./.test(name);
  let input = 0;
  let rest = 0;
  for (const t of tensors) {
    if (isInput(t.name)) input += t.size;
    else rest += t.size;
  }
  // rough size of the compute buffer of the output layer, it grows with the vocab
  const output = nVocab * nUbatch * 4 * 3;
  if (input + rest + output < RPC_BUDGET.single) return 0;
  // without output.weight, llama.cpp copies token_embd to the device of the output layer
  if (!tensors.some((t) => t.name === 'output.weight')) {
    rest += tensors.find((t) => t.name === 'token_embd.weight')?.size ?? 0;
  }
  const n = Math.ceil((rest + output) / RPC_BUDGET.worker);
  return Math.min(n, MAX_RPC_WORKERS);
};
