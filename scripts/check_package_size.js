#!/usr/bin/env node

import { execSync } from 'child_process';
import { existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
if (existsSync(resolve(__dirname, '../IS_DEBUG_BUILD'))) {
  console.error('ERROR: IS_DEBUG_BUILD file exists - this is a debug build and cannot be published');
  process.exit(1);
}

const MAX_SIZE = 20 * 1024 * 1024; // 20 MB
const MAX_FILES = 90;

const result = JSON.parse(execSync('npm pack --dry-run --json 2>/dev/null'));
const { unpackedSize, entryCount } = result[0];

console.log(`Unpacked size: ${(unpackedSize / 1024 / 1024).toFixed(2)} MB`);
console.log(`Total files: ${entryCount}`);

if (unpackedSize > MAX_SIZE) {
  console.error(`ERROR: Unpacked size exceeds 20 MB limit`);
  process.exit(1);
}

if (entryCount > MAX_FILES) {
  console.error(`ERROR: Total files (${entryCount}) exceeds limit of ${MAX_FILES}`);
  process.exit(1);
}
