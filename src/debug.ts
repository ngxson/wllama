import { WASM_SOURCE_MAP } from './wasm/source-map';

interface DecodedMap {
  firstId: number;
  funcNames: (string | null)[]; // indexed by (funcId - firstId)
}

const cache = new Map<string, DecodedMap>();

async function loadMap(buildKey: string): Promise<DecodedMap> {
  if (cache.has(buildKey)) return cache.get(buildKey)!;

  const b64 = WASM_SOURCE_MAP[buildKey];
  if (!b64) throw new Error(`No source map for build "${buildKey}"`);

  const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const ds = new DecompressionStream('gzip');
  const writer = ds.writable.getWriter();
  writer.write(gzipped);
  writer.close();
  const buf = await new Response(ds.readable).arrayBuffer();
  const dv = new DataView(buf);
  const bytes = new Uint8Array(buf);

  const firstId = dv.getUint32(0, true);
  const funcCount = dv.getUint32(4, true);
  const numNames = dv.getUint32(8, true);

  // Read name table
  const td = new TextDecoder();
  const names: string[] = [];
  let pos = 12;
  for (let i = 0; i < numNames; i++) {
    const len = bytes[pos++];
    names.push(td.decode(bytes.subarray(pos, pos + len)));
    pos += len;
  }

  // Read u16 index array
  const funcNames: (string | null)[] = [];
  for (let i = 0; i < funcCount; i++) {
    const idx = dv.getUint16(pos, true);
    pos += 2;
    funcNames.push(idx === 0xffff ? null : names[idx]);
  }

  const entry: DecodedMap = { firstId, funcNames };
  cache.set(buildKey, entry);
  return entry;
}

export const Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: async (
    funcIds: number[],
    isCompatBuild: boolean
  ): Promise<{ funcId: number; name: string }[]> => {
    const buildKey = isCompatBuild ? 'compat' : 'default';
    const { firstId, funcNames } = await loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name =
        i >= 0 && i < funcNames.length && funcNames[i]
          ? funcNames[i]!
          : '(unknown)';
      return { funcId, name };
    });
  },
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: async (
    stack: string,
    isCompatBuild: boolean
  ): Promise<string> => {
    // match wasm-function[N] from Chrome, Firefox and Safari stack formats
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1]))),
    ];
    if (funcIds.length === 0) return stack;

    const resolved = await Debug.decodeFuncIds(funcIds, isCompatBuild);

    return resolved
      .map((r) => {
        if (r.name === '(unknown)') {
          return `    wasm-func[${r.funcId}] (unknown)`;
        }
        return `    wasm-func[${r.funcId}] (${r.name})`;
      })
      .join('\n');
  },
};
