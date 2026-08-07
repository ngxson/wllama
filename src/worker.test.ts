import { expect, test } from 'vitest';
import { WllamaRuntimeError } from './wllama';
import { ProxyToWorker } from './worker';

test('reports worker calls after termination as runtime errors', async () => {
  const proxy = new ProxyToWorker(
    { wasmPath: '/wllama.wasm', compat: true },
    0,
    false,
    console
  );

  await proxy.wllamaExit();

  const debugCall = proxy.wllamaDebug();
  await expect(debugCall).rejects.toBeInstanceOf(WllamaRuntimeError);
  await expect(debugCall).rejects.toThrow('Wllama worker was terminated');

  const initCall = proxy.moduleInit([]);
  await expect(initCall).rejects.toBeInstanceOf(WllamaRuntimeError);
  await expect(initCall).rejects.toThrow('Wllama worker was terminated');
});

test('does not revive a worker after termination during module loading', async () => {
  const proxy = new ProxyToWorker(
    { wasmPath: '/wllama.wasm', compat: false },
    0,
    false,
    console
  );
  let finishModuleLoading!: (code: string) => void;
  proxy.getModuleCode = () =>
    new Promise((resolve) => {
      finishModuleLoading = resolve;
    });

  const initCall = proxy.moduleInit([]);
  await proxy.wllamaExit();
  finishModuleLoading('var Module = {};');

  await expect(initCall).rejects.toBeInstanceOf(WllamaRuntimeError);
  expect(proxy.worker).toBeUndefined();
});
