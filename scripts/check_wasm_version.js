#!/usr/bin/env node

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const DEFAULT_WASM = 'src/wasm/wllama.wasm';
const COMPAT_WASM = 'compat/wasm/wllama.wasm';

function fail(message) {
  console.error(`ERROR: ${message}`);
  process.exit(1);
}

// compare the commit only, the build number is wrong in a shallow clone
const getCommit = (version) => version.split('-').pop();

function checkFile(relPath, expected) {
  const absPath = resolve(ROOT, relPath);
  if (!existsSync(absPath)) fail(`${relPath} is missing`);
  const match = readFileSync(absPath, 'latin1').match(/"version":"(b\d+-[0-9a-f]+)"/);
  if (!match) fail(`${relPath} has no version, rebuild it with: npm run build:wasm`);
  if (getCommit(match[1]) !== getCommit(expected)) {
    fail(`${relPath} version is ${match[1]} but generated.ts has ${expected}, run: npm run build:wasm`);
  }
  console.log(`${relPath}: ${match[1]}`);
}

export function checkWasmVersion({ requireCompat }) {
  const generated = readFileSync(resolve(ROOT, 'src/workers-code/generated.ts'), 'utf8');
  const expected = generated.match(/LIBLLAMA_VERSION = '([^']+)'/)?.[1];
  if (!expected) fail('LIBLLAMA_VERSION not found in generated.ts');
  checkFile(DEFAULT_WASM, expected);
  if (requireCompat || existsSync(resolve(ROOT, COMPAT_WASM))) {
    checkFile(COMPAT_WASM, expected);
  }
}

// Run directly when invoked as a script
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  checkWasmVersion({ requireCompat: false });
}
