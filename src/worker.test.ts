import { test, expect, describe } from 'vitest';
import { ProxyToWorker } from './worker';
import { WllamaRuntimeError } from './wllama';

const silentLogger = {
  debug: () => {},
  log: () => {},
  warn: () => {},
  error: () => {},
};

// Push a pending task, send a signal.abort message with the given reason,
// and return what the pending task was rejected with.
const rejectionForAbort = (reason: unknown) => {
  const proxy = new ProxyToWorker(
    { wasmPath: '', compat: false },
    1,
    true,
    silentLogger
  );
  const pending = new Promise((resolve, reject) => {
    proxy.resultQueue.push({
      resolve,
      reject,
      param: { verb: 'load', args: [], callbackId: 1 },
    } as any);
  });
  (proxy as any).onRecvMsg({
    data: { verb: 'signal.abort', args: ['abort', reason, 'a|b', null] },
  });
  return pending.then(
    () => {
      throw new Error('task was not rejected');
    },
    (err) => err
  );
};

describe('signal.abort', () => {
  test('rejects pending tasks with a string reason', async () => {
    const err = await rejectionForAbort(
      'Aborted(oops). Build with -sASSERTIONS for more info.'
    );
    expect(err).toBeInstanceOf(WllamaRuntimeError);
    expect(err.message).toBe('(ABORT) Aborted(oops). ');
  });

  test('rejects pending tasks with an Error reason', async () => {
    const err = await rejectionForAbort(
      new WebAssembly.CompileError('Memory64 is not enabled')
    );
    expect(err).toBeInstanceOf(WllamaRuntimeError);
    expect(err.message).toBe('(ABORT) Memory64 is not enabled');
  });

  test('rejects pending tasks with a missing reason', async () => {
    const err = await rejectionForAbort(undefined);
    expect(err).toBeInstanceOf(WllamaRuntimeError);
    expect(err.message).toBe('(ABORT) ');
  });
});
