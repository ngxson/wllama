// A fake llama GGUF, split into shards, made in memory: the header is built once, the tensor data is made when a range is read.
// Weights are Q8_0 with a small scale and random int8, norms are 1.0, so all values are finite.

const GGUF_MAGIC = 0x46554747; // "GGUF"
const ALIGNMENT = 32;
const T_U16 = 2;
const T_U32 = 4;
const T_I32 = 5;
const T_F32 = 6;
const T_STRING = 8;
const T_ARRAY = 9;
const GGML_F32 = 0;
const GGML_Q8_0 = 8;

export interface FakeModelParams {
  nEmbd: number;
  nFf: number;
  nLayer: number;
  nHead: number;
  shardSize: number; // target data size of one shard
}

interface FakeTensor {
  name: string;
  ne: number[];
  type: number;
  size: number;
}

class Writer {
  private parts: Uint8Array[] = [];
  length = 0;
  bytes(b: Uint8Array) {
    this.parts.push(b);
    this.length += b.byteLength;
  }
  private num(size: number, write: (d: DataView) => void) {
    const b = new Uint8Array(size);
    write(new DataView(b.buffer));
    this.bytes(b);
  }
  u16(v: number) {
    this.num(2, (d) => d.setUint16(0, v, true));
  }
  u32(v: number) {
    this.num(4, (d) => d.setUint32(0, v, true));
  }
  i32(v: number) {
    this.num(4, (d) => d.setInt32(0, v, true));
  }
  f32(v: number) {
    this.num(4, (d) => d.setFloat32(0, v, true));
  }
  u64(v: number) {
    this.num(8, (d) => d.setBigUint64(0, BigInt(v), true));
  }
  str(s: string) {
    const b = new TextEncoder().encode(s);
    this.u64(b.byteLength);
    this.bytes(b);
  }
  kv(key: string, type: number, write: () => void) {
    this.str(key);
    this.u32(type);
    write();
  }
  array(key: string, itemType: number, items: any[], write: (v: any) => void) {
    this.kv(key, T_ARRAY, () => {
      this.u32(itemType);
      this.u64(items.length);
      items.forEach(write);
    });
  }
  // padded with zeros to the alignment
  finish(): Uint8Array {
    const out = new Uint8Array(Math.ceil(this.length / ALIGNMENT) * ALIGNMENT);
    let pos = 0;
    for (const p of this.parts) {
      out.set(p, pos);
      pos += p.byteLength;
    }
    return out;
  }
}

const pad = (n: number) => Math.ceil(n / ALIGNMENT) * ALIGNMENT;

// repeated in the Q8_0 tensors: blocks of a fp16 scale (0x1C00 = 1/256) and 32 random int8
const Q8_BLOCK = 34;
const Q8_PATTERN = (() => {
  const p = new Uint8Array(Q8_BLOCK * 1024);
  for (let b = 0; b < 1024; b++) {
    p[b * Q8_BLOCK] = 0x00;
    p[b * Q8_BLOCK + 1] = 0x1c;
    for (let i = 2; i < Q8_BLOCK; i++) {
      p[b * Q8_BLOCK + i] = Math.floor(Math.random() * 255) - 127;
    }
  }
  return p;
})();
const F32_ONE = new Uint8Array([0, 0, 0x80, 0x3f]);

// copy `pattern`, read from position `from` (wrapping), into `dest`
const fillPattern = (dest: Uint8Array, pattern: Uint8Array, from: number) => {
  let pos = 0;
  let p = from % pattern.byteLength;
  while (pos < dest.byteLength) {
    const n = Math.min(pattern.byteLength - p, dest.byteLength - pos);
    dest.set(pattern.subarray(p, p + n), pos);
    pos += n;
    p = 0;
  }
};

/**
 * Looks like a Blob for wllama: only `size`, `name`, `slice()` and `arrayBuffer()` are used
 */
class FakeShard {
  constructor(
    public name: string,
    public size: number,
    private header: Uint8Array,
    private tensors: { offset: number; t: FakeTensor }[], // offset in the file
    private start = 0
  ) {}
  slice(a = 0, b = this.size) {
    b = Math.min(b, this.size);
    const s = new FakeShard(
      this.name,
      Math.max(0, b - a),
      this.header,
      this.tensors,
      this.start + a
    );
    return s;
  }
  async arrayBuffer() {
    const out = new Uint8Array(this.size);
    const end = this.start + this.size;
    // header
    if (this.start < this.header.byteLength) {
      const n = Math.min(this.header.byteLength, end) - this.start;
      out.set(this.header.subarray(this.start, this.start + n), 0);
    }
    // tensor data, the padding stays zero
    for (const { offset, t } of this.tensors) {
      const a = Math.max(offset, this.start);
      const b = Math.min(offset + t.size, end);
      if (a >= b) continue;
      const pattern = t.type === GGML_Q8_0 ? Q8_PATTERN : F32_ONE;
      fillPattern(
        out.subarray(a - this.start, b - this.start),
        pattern,
        a - offset
      );
    }
    return out.buffer;
  }
}

const tensorSize = (ne: number[], type: number) => {
  const n = ne.reduce((a, b) => a * b, 1);
  return type === GGML_Q8_0 ? (n / 32) * Q8_BLOCK : n * 4;
};

/**
 * A fake llama model split into shards of about `shardSize`, as Blob-like objects named model-0000k-of-0000n.gguf
 */
export const makeFakeModel = (p: FakeModelParams): Blob[] => {
  // vocab: unk, bos, eos, the SPM space and the printable ASCII chars, so any output is valid UTF-8 (byte tokens are not)
  const tokens = ['<unk>', '<s>', '</s>', '\u2581'];
  const types = [2, 3, 3, 1]; // unknown, control, control, normal
  for (let c = 33; c < 127; c++) {
    tokens.push(String.fromCharCode(c));
    types.push(1);
  }
  const nVocab = tokens.length;

  const t = (name: string, ne: number[], type = GGML_Q8_0): FakeTensor => ({
    name,
    ne,
    type,
    size: tensorSize(ne, type),
  });
  const { nEmbd, nFf } = p;
  const tensors: FakeTensor[] = [
    t('token_embd.weight', [nEmbd, nVocab]),
    t('output_norm.weight', [nEmbd], GGML_F32),
    t('output.weight', [nEmbd, nVocab]),
  ];
  for (let i = 0; i < p.nLayer; i++) {
    tensors.push(
      t(`blk.${i}.attn_norm.weight`, [nEmbd], GGML_F32),
      t(`blk.${i}.attn_q.weight`, [nEmbd, nEmbd]),
      t(`blk.${i}.attn_k.weight`, [nEmbd, nEmbd]),
      t(`blk.${i}.attn_v.weight`, [nEmbd, nEmbd]),
      t(`blk.${i}.attn_output.weight`, [nEmbd, nEmbd]),
      t(`blk.${i}.ffn_norm.weight`, [nEmbd], GGML_F32),
      t(`blk.${i}.ffn_gate.weight`, [nEmbd, nFf]),
      t(`blk.${i}.ffn_down.weight`, [nFf, nEmbd]),
      t(`blk.${i}.ffn_up.weight`, [nEmbd, nFf])
    );
  }

  // group the tensors into shards
  const groups: FakeTensor[][] = [[]];
  let groupSize = 0;
  for (const tensor of tensors) {
    if (groupSize > 0 && groupSize + pad(tensor.size) > p.shardSize) {
      groups.push([]);
      groupSize = 0;
    }
    groups[groups.length - 1].push(tensor);
    groupSize += pad(tensor.size);
  }

  return groups.map((group, k) => {
    const w = new Writer();
    const splitKeys = () => {
      w.kv('split.no', T_U16, () => w.u16(k));
      w.kv('split.count', T_U16, () => w.u16(groups.length));
      w.kv('split.tensors.count', T_I32, () => w.i32(tensors.length));
    };
    const kvs: (() => void)[] = [splitKeys];
    if (k === 0) {
      const u32 = (key: string, v: number) => () =>
        w.kv(key, T_U32, () => w.u32(v));
      kvs.push(
        () => w.kv('general.architecture', T_STRING, () => w.str('llama')),
        () => w.kv('general.name', T_STRING, () => w.str('fake')),
        u32('llama.context_length', 2048),
        u32('llama.embedding_length', nEmbd),
        u32('llama.block_count', p.nLayer),
        u32('llama.feed_forward_length', nFf),
        u32('llama.attention.head_count', p.nHead),
        u32('llama.attention.head_count_kv', p.nHead),
        u32('llama.rope.dimension_count', nEmbd / p.nHead),
        () =>
          w.kv('llama.attention.layer_norm_rms_epsilon', T_F32, () =>
            w.f32(1e-5)
          ),
        () => w.kv('tokenizer.ggml.model', T_STRING, () => w.str('llama')),
        () => w.array('tokenizer.ggml.tokens', T_STRING, tokens, w.str.bind(w)),
        () =>
          w.array(
            'tokenizer.ggml.scores',
            T_F32,
            tokens.map(() => 0),
            (v) => w.f32(v)
          ),
        () =>
          w.array('tokenizer.ggml.token_type', T_I32, types, (v) => w.i32(v)),
        u32('tokenizer.ggml.unknown_token_id', 0),
        u32('tokenizer.ggml.bos_token_id', 1),
        u32('tokenizer.ggml.eos_token_id', 2)
      );
    }
    w.u32(GGUF_MAGIC);
    w.u32(3);
    w.u64(group.length);
    w.u64(kvs.length + 2); // the split keys are 3 KVs
    kvs.forEach((write) => write());
    const placed: { offset: number; t: FakeTensor }[] = [];
    let dataOffset = 0;
    for (const tensor of group) {
      w.str(tensor.name);
      w.u32(tensor.ne.length);
      tensor.ne.forEach((n) => w.u64(n));
      w.u32(tensor.type);
      w.u64(dataOffset);
      placed.push({ offset: dataOffset, t: tensor });
      dataOffset += pad(tensor.size);
    }
    const header = w.finish();
    const name = `fake-${String(k + 1).padStart(5, '0')}-of-${String(groups.length).padStart(5, '0')}.gguf`;
    const inFile = placed.map(({ offset, t }) => ({
      offset: header.byteLength + offset,
      t,
    }));
    return new FakeShard(
      name,
      header.byteLength + dataOffset,
      header,
      inFile
    ) as unknown as Blob;
  });
};
