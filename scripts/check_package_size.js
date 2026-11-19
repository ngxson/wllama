#!/usr/bin/env node

import { execSync } from 'child_process';
import { checkDebugBuild } from './check_debug_build.js';
import { checkWasmVersion } from './check_wasm_version.js';

checkDebugBuild();
checkWasmVersion({ requireCompat: true });

const MAX_SIZE = 20 * 1024 * 1024; // 20 MB
const MAX_FILES = 90;

function checkPackage(dir) {
  const result = JSON.parse(execSync('npm pack --dry-run --json 2>/dev/null', { cwd: dir }));
  const { name, unpackedSize, entryCount, files } = result[0];

  console.log(`${name}: unpacked size: ${(unpackedSize / 1024 / 1024).toFixed(2)} MB`);
  console.log(`${name}: total files: ${entryCount}`);

  if (unpackedSize > MAX_SIZE) {
    console.error(`ERROR: ${name}: unpacked size exceeds 20 MB limit`);
    process.exit(1);
  }

  if (entryCount > MAX_FILES) {
    console.error(files);
    console.error(`ERROR: ${name}: total files (${entryCount}) exceeds limit of ${MAX_FILES}`);
    process.exit(1);
  }
}

checkPackage(new URL('../', import.meta.url));
checkPackage(new URL('../compat/', import.meta.url));
