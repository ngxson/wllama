import { expect, test } from 'vitest';
import { WLLAMA_EMSCRIPTEN_CODE } from './workers-code/generated';

const WASM_PAGE_BYTES = 64 * 1024;
const FOUR_GIB_BYTES = 4 * 1024 * 1024 * 1024;
const MEMORY64_MAX_BYTES = 16 * 1024 * 1024 * 1024;
const MEMORY64_MAX_PAGES = MEMORY64_MAX_BYTES / WASM_PAGE_BYTES;
const IS_CHROMIUM = /Chrome\//.test(navigator.userAgent);

// Firefox 150 runs ordinary inference with this Memory64 artifact, but stalls
// for over 60 seconds while materializing the probe's full 16 GiB buffer.
// Keep the extreme sparse-growth check on Chromium; the Firefox suite still
// exercises real model loading and inference against the default artifact.

// This module imports Wllama's shared Memory64 memory and reads or writes a
// byte at an i64 address. It is equivalent to:
//
// (module
//   (import "env" "memory" (memory i64 2048 262144 shared))
//   (func (export "write") (param i64 i32)
//     local.get 0 local.get 1 i32.store8)
//   (func (export "read") (param i64) (result i32)
//     local.get 0 i32.load8_u))
const MEMORY64_PROBE_BYTES = [
  0, 97, 115, 109, 1, 0, 0, 0, 1, 11, 2, 96, 2, 126, 127, 0, 96, 1, 126, 1, 127,
  2, 19, 1, 3, 101, 110, 118, 6, 109, 101, 109, 111, 114, 121, 2, 7, 128, 16,
  128, 128, 16, 3, 3, 2, 0, 1, 7, 16, 2, 5, 119, 114, 105, 116, 101, 0, 0, 4,
  114, 101, 97, 100, 0, 1, 10, 19, 2, 9, 0, 32, 0, 32, 1, 58, 0, 0, 11, 7, 0,
  32, 0, 45, 0, 0, 11,
];

interface Memory64ProbeResult {
  initialBytes: number;
  boundaryBytes: number;
  belowBoundary: number;
  aboveBoundary: number;
  finalBytes: number;
  finalByte: number;
}

const runMemory64Probe = async (): Promise<Memory64ProbeResult> => {
  const wasmUrl = new URL('/src/wasm/wllama.wasm', location.href).href;
  const workerCode = `
    var Module = {
      noInitialRun: true,
      pthreadPoolSize: 0,
      locateFile: () => ${JSON.stringify(wasmUrl)},
      onRuntimeInitialized: () => {
        try {
          const pageBytes = ${WASM_PAGE_BYTES};
          const fourGiBBytes = ${FOUR_GIB_BYTES};
          const maximumBytes = ${MEMORY64_MAX_BYTES};
          const maximumPages = ${MEMORY64_MAX_PAGES};
          const memory = Module.wasmMemory;
          const initialBytes = memory.buffer.byteLength;
          const currentPages = BigInt(initialBytes / pageBytes);
          const boundaryPages = BigInt(fourGiBBytes / pageBytes + 1);

          memory.grow(boundaryPages - currentPages);

          const probeModule = new WebAssembly.Module(
            new Uint8Array(${JSON.stringify(MEMORY64_PROBE_BYTES)})
          );
          const probe = new WebAssembly.Instance(probeModule, {
            env: { memory },
          }).exports;
          const belowBoundaryAddress = BigInt(fourGiBBytes - 1);
          const aboveBoundaryAddress = BigInt(fourGiBBytes);

          probe.write(belowBoundaryAddress, 0x2a);
          probe.write(aboveBoundaryAddress, 0x2b);

          const boundaryBytes = memory.buffer.byteLength;
          const belowBoundary = probe.read(belowBoundaryAddress);
          const aboveBoundary = probe.read(aboveBoundaryAddress);
          const grownPages = BigInt(boundaryBytes / pageBytes);

          memory.grow(BigInt(maximumPages) - grownPages);

          const finalAddress = BigInt(maximumBytes - 1);
          probe.write(finalAddress, 0x2c);

          postMessage({
            initialBytes,
            boundaryBytes,
            belowBoundary,
            aboveBoundary,
            finalBytes: memory.buffer.byteLength,
            finalByte: probe.read(finalAddress),
          });
        } catch (error) {
          postMessage({
            error: error instanceof Error ? error.message : String(error),
          });
        }
      },
    };

    ${WLLAMA_EMSCRIPTEN_CODE}
  `;
  const workerUrl = URL.createObjectURL(
    new Blob([workerCode], { type: 'text/javascript' })
  );
  const worker = new Worker(workerUrl);

  try {
    return await new Promise<Memory64ProbeResult>((resolve, reject) => {
      const timeout = window.setTimeout(
        () => reject(new Error('Memory64 browser probe timed out')),
        60_000
      );

      worker.onmessage = ({ data }) => {
        window.clearTimeout(timeout);
        if (data.error) reject(new Error(data.error));
        else resolve(data);
      };
      worker.onerror = ({ message }) => {
        window.clearTimeout(timeout);
        reject(new Error(message));
      };
    });
  } finally {
    worker.terminate();
    URL.revokeObjectURL(workerUrl);
  }
};

test
  .runIf(IS_CHROMIUM)
  .sequential(
    'the default artifact addresses memory above 4 GiB and through 16 GiB',
    async () => {
      const result = await runMemory64Probe();

      expect(result.initialBytes).toBe(128 * 1024 * 1024);
      expect(result.boundaryBytes).toBe(FOUR_GIB_BYTES + WASM_PAGE_BYTES);
      expect(result.belowBoundary).toBe(0x2a);
      expect(result.aboveBoundary).toBe(0x2b);
      expect(result.finalBytes).toBe(MEMORY64_MAX_BYTES);
      expect(result.finalByte).toBe(0x2c);
    }
  );
