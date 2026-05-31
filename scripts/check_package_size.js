#!/usr/bin/env node

import { execSync } from 'child_process';
import { checkDebugBuild } from './check_debug_build.js';

checkDebugBuild();

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
