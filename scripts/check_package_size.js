#!/usr/bin/env node

import { execSync } from 'child_process';

const MAX_SIZE = 20 * 1024 * 1024; // 20 MB

const result = JSON.parse(execSync('npm pack --dry-run --json 2>/dev/null'));
const size = result[0].unpackedSize;

console.log(`Unpacked size: ${(size / 1024 / 1024).toFixed(2)} MB`);

if (size > MAX_SIZE) {
  console.error(`ERROR: Unpacked size exceeds 20 MB limit`);
  process.exit(1);
}
