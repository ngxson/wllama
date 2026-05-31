#!/usr/bin/env node

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const WASM_PATHS = [
  'src/wasm/wllama.wasm',
  'compat/src/wasm/wllama.wasm',
];
const DEBUG_MARKER = 'test-backend-ops.cpp';

export function checkDebugBuild() {
  for (const relPath of WASM_PATHS) {
    const absPath = resolve(ROOT, relPath);
    if (!existsSync(absPath)) continue;
    const contents = readFileSync(absPath);
    if (contents.includes(DEBUG_MARKER)) {
      console.error(`ERROR: ${relPath} contains "${DEBUG_MARKER}" - this is a debug build and cannot be merged to master or be published`);
      process.exit(1);
    }
  }
}

// Run directly when invoked as a script
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  checkDebugBuild();
}
