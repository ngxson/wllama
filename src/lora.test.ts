import { describe, expect, it } from 'vitest';
import { baseLoraSelection, loraFileName, prepareBlobs } from './utils';

describe('LoRA blob staging', () => {
  it('uses deterministic adapter paths', () => {
    expect(loraFileName(0)).toBe('lora-00001.gguf');
    expect(loraFileName(11)).toBe('lora-00012.gguf');
  });

  it('mounts adapter blobs without treating them as model shards', async () => {
    const model = new Blob([new Uint8Array([0x47, 0x47, 0x55, 0x46])]);
    const adapterA = new Blob(['adapter-a']);
    const adapterB = new Blob(['adapter-b']);
    const prepared = await prepareBlobs([model], [adapterA, adapterB]);

    expect(prepared.llm).toHaveLength(1);
    expect(prepared.lora.map((entry) => entry.name)).toEqual([
      'lora-00001.gguf',
      'lora-00002.gguf',
    ]);
    expect(prepared.all).toHaveLength(3);
    expect(prepared.llm.some((entry) => entry.name.startsWith('lora-'))).toBe(false);
  });
});

describe('LoRA request semantics', () => {
  it('creates an explicit base-weight selection', () => {
    expect(baseLoraSelection(3)).toEqual([
      { id: 0, scale: 0 },
      { id: 1, scale: 0 },
      { id: 2, scale: 0 },
    ]);
  });
});
