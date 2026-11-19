#!/usr/bin/env node
// Usage: node scripts/pack_wasm.js <input.wasm> <output.wasm>
// Pack the emscripten output into one wasm file: add the JS glue and the function names as custom sections. See README-dev.md

import { existsSync, readFileSync, writeFileSync } from 'fs';

// keep in sync with src/worker.ts
const JS_SECTION_NAME = 'wllama_js';

const [inputFile, outputFile] = process.argv.slice(2);
if (!inputFile || !outputFile) {
  console.error('Usage: node scripts/pack_wasm.js <input.wasm> <output.wasm>');
  process.exit(1);
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

// -- wasm encoder -------------------------------------------------------------

function encodeUleb(value) {
  const bytes = [];
  do {
    let b = value & 0x7F;
    value >>>= 7;
    if (value) b |= 0x80;
    bytes.push(b);
  } while (value);
  return Buffer.from(bytes);
}

function encodeString(str) {
  const b = Buffer.from(str);
  return Buffer.concat([encodeUleb(b.length), b]);
}

function encodeNameSection(symbols) {
  const FUNCTION_NAMES = 1;
  const entries = [];
  let count = 0;
  for (const [id, raw] of symbols) {
    const name = simplifyName(raw);
    if (!name) continue;
    entries.push(encodeUleb(id), encodeString(name));
    count++;
  }
  const names = Buffer.concat([encodeUleb(count), ...entries]);
  const body = Buffer.concat([
    encodeString('name'),
    Buffer.from([FUNCTION_NAMES]),
    encodeUleb(names.length),
    names,
  ]);
  return encodeCustomSection(body);
}

function encodeCustomSection(body) {
  const CUSTOM_SECTION = 0;
  return Buffer.concat([Buffer.from([CUSTOM_SECTION]), encodeUleb(body.length), body]);
}

function encodeDataSection(name, data) {
  return encodeCustomSection(Buffer.concat([encodeString(name), data]));
}

// -- wasm decoder -------------------------------------------------------------

function decodeUleb(buf, pos) {
  let value = 0, shift = 0, b;
  do {
    b = buf[pos++];
    value += (b & 0x7F) * 2 ** shift;
    shift += 7;
  } while (b & 0x80);
  return [value, pos];
}

// list the names of all custom sections
function listCustomSections(wasm) {
  const names = [];
  let pos = 8; // skip magic + version
  while (pos < wasm.length) {
    const id = wasm[pos++];
    let size;
    [size, pos] = decodeUleb(wasm, pos);
    if (id === 0) {
      const [len, p] = decodeUleb(wasm, pos);
      names.push(wasm.subarray(p, p + len).toString());
    }
    pos += size;
  }
  return names;
}

// -- main ---------------------------------------------------------------------

const wasm = readFileSync(inputFile);
const existingSections = listCustomSections(wasm);
if (existingSections.includes(JS_SECTION_NAME)) {
  writeFileSync(outputFile, wasm);
  process.stderr.write(`${outputFile}: already packed, copied as-is\n`);
  process.exit(0);
}

const sections = [];

// the name section needs ascending function indices
const symbolsFile = inputFile.replace(/\.wasm$/, '.js.symbols');
if (existingSections.includes('name')) {
  process.stderr.write(`${outputFile}: already has a name section, skipped\n`);
} else if (!existsSync(symbolsFile)) {
  process.stderr.write(`${outputFile}: ${symbolsFile} not found, name section skipped\n`);
} else {
  const symbols = readFileSync(symbolsFile, 'utf8')
    .split('\n')
    .map((line) => {
      const colon = line.indexOf(':');
      return [parseInt(line.slice(0, colon)), line.slice(colon + 1).trim()];
    })
    .filter(([id]) => !isNaN(id))
    .sort(([a], [b]) => a - b);
  const section = encodeNameSection(symbols);
  sections.push(section);
  process.stderr.write(`${outputFile}: ${symbols.length} symbols, name section is ${section.length.toLocaleString()} bytes\n`);
}

const jsFile = inputFile.replace(/\.wasm$/, '.js');
const js = readFileSync(jsFile);
const jsSection = encodeDataSection(JS_SECTION_NAME, js);
sections.push(jsSection);
process.stderr.write(`${outputFile}: embedded ${jsFile}, ${JS_SECTION_NAME} section is ${jsSection.length.toLocaleString()} bytes\n`);

writeFileSync(outputFile, Buffer.concat([wasm, ...sections]));
