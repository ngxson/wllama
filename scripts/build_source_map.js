#!/usr/bin/env node
// Usage: node scripts/build_source_map.js [--input name:buildDir] [--output out.ts]
// All flags are optional — defaults point to build/ and src/wasm/source-map.ts.
// Reads *.js.symbols from each build dir and produces cleaned function names.

import { readFileSync, writeFileSync, readdirSync } from 'fs';
import { resolve, join } from 'path';
import { gzipSync } from 'zlib';

const DEFAULT_INPUTS = [
  { name: 'default', symbolsPath: 'build' },
  { name: 'compat',  symbolsPath: 'build-compat' },
];
const DEFAULT_OUTPUT = 'src/wasm/source-map.ts';

const args = process.argv.slice(2);
const inputs = [];
let outputFile = null;

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--input') {
    const next = args[++i];
    if (!next) { console.error('--input must be name:buildDir'); process.exit(1); }
    const [name, symbolsPath] = next.split(':');
    if (!name || !symbolsPath) { console.error('--input must be name:buildDir'); process.exit(1); }
    inputs.push({ name, symbolsPath: resolve(symbolsPath) });
  } else if (args[i] === '--output') {
    const next = args[++i];
    if (!next) { console.error('--output requires a path'); process.exit(1); }
    outputFile = resolve(next);
  }
}

if (!inputs.length) {
  for (const { name, symbolsPath } of DEFAULT_INPUTS) {
    const resolved = resolve(symbolsPath);
    try { readdirSync(resolved); inputs.push({ name, symbolsPath: resolved }); } catch { /* skip */ }
  }
}
if (!outputFile) outputFile = resolve(DEFAULT_OUTPUT);
if (!inputs.length) { console.error('No build directories found.'); process.exit(1); }

// -- wasm binary: extract [firstFuncId, funcCount] ----------------------------

function readUleb(buf, pos) {
  let result = 0, shift = 0;
  while (true) {
    const b = buf[pos++];
    result |= (b & 0x7F) << shift;
    if (!(b & 0x80)) break;
    shift += 7;
  }
  return [result, pos];
}

function skipLimits(buf, pos) {
  const kind = buf[pos++];
  [, pos] = readUleb(buf, pos);
  if (kind === 0x01 || kind === 0x03 || kind === 0x05 || kind === 0x07)
    [, pos] = readUleb(buf, pos);
  return pos;
}

function parseFuncIds(wasmBuf) {
  let pos = 8, importCount = 0;
  while (pos < wasmBuf.length) {
    const sectionId = wasmBuf[pos++];
    let sectionSize;
    [sectionSize, pos] = readUleb(wasmBuf, pos);
    const sectionEnd = pos + sectionSize;
    if (sectionId === 2) {
      let count; [count, pos] = readUleb(wasmBuf, pos);
      for (let i = 0; i < count; i++) {
        let nl;
        [nl, pos] = readUleb(wasmBuf, pos); pos += nl;
        [nl, pos] = readUleb(wasmBuf, pos); pos += nl;
        const kind = wasmBuf[pos++];
        if (kind === 0)      { [, pos] = readUleb(wasmBuf, pos); importCount++; }
        else if (kind === 1) { pos++; pos = skipLimits(wasmBuf, pos); }
        else if (kind === 2) { pos = skipLimits(wasmBuf, pos); }
        else if (kind === 3) { pos += 2; }
      }
    } else if (sectionId === 10) {
      let funcCount; [funcCount, pos] = readUleb(wasmBuf, pos);
      return [importCount, funcCount];
    } else {
      pos = sectionEnd;
    }
  }
  throw new Error('No code section found');
}

// -- name simplification ------------------------------------------------------

function truncateTemplates(name, maxLen) {
  let result = '', i = 0;
  while (i < name.length) {
    if (name[i] !== '<') { result += name[i++]; continue; }
    // find matching >
    let depth = 1, j = i + 1;
    while (j < name.length && depth > 0) {
      if (name[j] === '<') depth++;
      else if (name[j] === '>') depth--;
      j++;
    }
    const content = name.slice(i + 1, j - 1);
    result += content.length > maxLen
      ? '<' + content.slice(0, maxLen) + '...>'
      : '<' + content + '>';
    i = j;
  }
  return result;
}

function stripParams(name) {
  // Find first top-level '(' (not inside <>) and strip from there.
  // If params are empty "()", keep them; otherwise drop entirely.
  let depth = 0;
  for (let i = 0; i < name.length; i++) {
    const c = name[i];
    if (c === '<') { depth++; continue; }
    if (c === '>') { depth--; continue; }
    if (c === '(' && depth === 0) {
      const base = name.slice(0, i);
      return (name[i + 1] === ')') ? base + '()' : base;
    }
  }
  return name;
}

const STD_HINT = 'std::...';

function simplifyName(raw) {
  if (!raw) return null;
  let name = raw;

  // Rule 0: collapse all std:: into a hint
  if (/^std::/.test(name)) return STD_HINT;

  // Rule 1: lambda / closure types
  // Handles ::$_N, ::'lambda'(), 'lambda'()
  // Strategy: find the marker, take everything inside the nearest enclosing <..> before it
  const lambdaRe = /::[$']_?\d*|::'lambda'/;
  const lambdaMatch = lambdaRe.exec(name);
  if (lambdaMatch) {
    const before = name.slice(0, lambdaMatch.index);
    // Extract the innermost meaningful context: look for the last '<' before the marker
    const lastAngle = before.lastIndexOf('<');
    let parent = lastAngle >= 0 ? before.slice(lastAngle + 1) : before;
    // Strip trailing qualifiers
    parent = parent.replace(/\s+(const|volatile|noexcept|&&?)\s*$/, '').trim();
    name = parent;
    // fall through to further cleanup below
  }

  // Rule 2: strip parameter list
  name = stripParams(name);

  // Rule 3: remove libc++ internal sub-namespaces (::__2::, ::__1::, etc.)
  name = name.replace(/::__\d+::/g, '::');

  // Rule 4: remove ABI tags
  name = name.replace(/\[abi:[^\]]+\]/g, '');

  // Rule 5: truncate template args to 10 chars
  name = truncateTemplates(name, 10);

  // Rule 6: final cleanup
  name = name.replace(/::::/g, '::').replace(/\s+/g, ' ').trim();

  return name || null;
}

// -- binary encoder -----------------------------------------------------------

function encodeNames(funcCount, firstId, symbols) {
  // Build deduplicated name table
  const nameToIdx = new Map(); // string -> u16 index
  const nameTable = [];        // array of Buffer
  const indices = new Uint16Array(funcCount); // 0xFFFF = unknown
  indices.fill(0xFFFF);

  let mapped = 0;
  for (let i = 0; i < funcCount; i++) {
    const raw = symbols.get(firstId + i) ?? null;
    const cleaned = raw ? simplifyName(raw) : null;
    if (!cleaned) continue;
    let idx = nameToIdx.get(cleaned);
    if (idx === undefined) {
      idx = nameTable.length;
      const b = Buffer.from(cleaned.slice(0, 254));
      nameTable.push(Buffer.concat([Buffer.from([b.length]), b]));
      nameToIdx.set(cleaned, idx);
    }
    indices[i] = idx;
    mapped++;
  }

  const numNames = nameTable.length;
  process.stderr.write(`    ${mapped}/${funcCount} named, ${numNames} unique names\n`);

  // u32 numNames + name table + u16 index array
  const header = Buffer.alloc(4);
  header.writeUInt32LE(numNames, 0);
  const indexBuf = Buffer.from(indices.buffer);
  return Buffer.concat([header, ...nameTable, indexBuf]);
}

// -- resolve symbolsPath (dir or file) ----------------------------------------

function resolveSymbolsFile(symbolsPath) {
  if (symbolsPath.endsWith('.js.symbols')) return symbolsPath;
  for (const entry of readdirSync(symbolsPath))
    if (entry.endsWith('.js.symbols')) return join(symbolsPath, entry);
  throw new Error(`No .js.symbols file in ${symbolsPath}`);
}

// -- per-build processing -----------------------------------------------------

function processBuild(symbolsPath) {
  const resolvedSymbols = resolveSymbolsFile(symbolsPath);
  const wasmPath = resolvedSymbols.replace(/\.js\.symbols$/, '.wasm');

  process.stderr.write(`  Parsing wasm binary...\n`);
  const [firstId, funcCount] = parseFuncIds(readFileSync(wasmPath));
  process.stderr.write(`    ${funcCount} functions starting at index ${firstId}\n`);

  process.stderr.write(`  Loading symbols...\n`);
  const symbols = new Map();
  for (const line of readFileSync(resolvedSymbols, 'utf8').split('\n')) {
    const colon = line.indexOf(':');
    if (colon < 0) continue;
    const id = parseInt(line.slice(0, colon));
    if (!isNaN(id)) symbols.set(id, line.slice(colon + 1).trim());
  }
  process.stderr.write(`    ${symbols.size} raw symbols\n`);

  const header = Buffer.alloc(8);
  header.writeUInt32LE(firstId, 0);
  header.writeUInt32LE(funcCount, 4);

  const nameData  = encodeNames(funcCount, firstId, symbols);
  const binary    = Buffer.concat([header, nameData]);
  const compressed = gzipSync(binary);
  process.stderr.write(`    ${binary.length.toLocaleString()} bytes -> ${compressed.length.toLocaleString()} bytes gzipped\n`);

  return compressed.toString('base64');
}

// -- main ---------------------------------------------------------------------

const entries_ts = [];
for (const { name, symbolsPath } of inputs) {
  process.stderr.write(`\n[${name}] ${symbolsPath}\n`);
  entries_ts.push(`  "${name}": "${processBuild(symbolsPath)}"`);
}

const tsContent = [
  `// Auto-generated by scripts/build_source_map.js — do not edit`,
  `// Format: gzip-compressed binary name table, base64-encoded`,
  `// Structure: u32 firstId, u32 funcCount, u32 numNames, then name table (u8 len + bytes each), then u16 index array (0xFFFF = unknown)`,
  `export const WASM_SOURCE_MAP: Record<string, string> = {`,
  entries_ts.join(',\n'),
  `};`,
  ``,
].join('\n');

writeFileSync(outputFile, tsContent);
process.stderr.write(`\nWrote ${outputFile}\n`);
