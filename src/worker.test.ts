import { expect, test } from 'vitest';
import { WllamaRuntimeError } from './wllama';
import { ProxyToWorker } from './worker';

test('reports calls after termination as runtime errors', async () => {
  const proxy = new ProxyToWorker(
    { wasmPath: '/wllama.wasm', compat: false },
    0,
    false,
    console
  );

  await proxy.wllamaExit();

  await expect(proxy.wllamaDebug()).rejects.toBeInstanceOf(WllamaRuntimeError);
  await expect(proxy.wllamaDebug()).rejects.toThrow(
    'Wllama worker was terminated'
  );
});
