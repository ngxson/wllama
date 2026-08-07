import assert from 'node:assert/strict';
import test from 'node:test';

import {
  closeAndCheckBrowserEvents,
  createFailureLatch,
  createSingleFlight,
  matchesExpectedCompletion,
  resolveCgroupMount,
} from './memory64_stress_helpers.mjs';

test('cgroup resolution selects a later applicable mount', () => {
  const mountInfo = [
    '155 145 0:32 /other /sys/fs/cgroup-other rw - cgroup2 cgroup rw',
    '156 145 0:33 /workload /sys/fs/cgroup rw - cgroup2 cgroup rw',
  ].join('\n');

  assert.deepEqual(resolveCgroupMount(mountInfo, 'cgroup2', '/workload/job'), {
    mountFound: true,
    path: '/sys/fs/cgroup/job',
  });
});

test('namespaced cgroup roots are marked unresolved instead of absent', () => {
  const mountInfo = '155 145 0:32 /.. /sys/fs/cgroup rw - cgroup2 cgroup rw';

  assert.deepEqual(resolveCgroupMount(mountInfo, 'cgroup2', '/'), {
    mountFound: true,
    path: null,
  });
});

test('an already received signal rejects work before it starts', async () => {
  const failureLatch = createFailureLatch();
  let started = false;

  failureLatch.fail('Stress run interrupted by SIGTERM');
  const beginWork = () => {
    failureLatch.throwIfFailed();
    started = true;
  };

  assert.throws(beginWork, /SIGTERM/);
  assert.equal(started, false);

  await assert.rejects(failureLatch.guard(Promise.resolve()), /SIGTERM/);
});

test('a signal during diagnostics prevents the following launch', async () => {
  const failureLatch = createFailureLatch();
  let finishDiagnostics;
  let launched = false;
  const diagnostics = new Promise((resolve) => {
    finishDiagnostics = resolve;
  });
  const run = (async () => {
    await failureLatch.guard(diagnostics);
    failureLatch.throwIfFailed();
    launched = true;
  })();

  failureLatch.fail('Stress run interrupted by SIGINT');
  finishDiagnostics();

  await assert.rejects(run, /SIGINT/);
  assert.equal(launched, false);
});

test('a fatal event during browser close rejects finalization', async () => {
  const failureLatch = createFailureLatch();

  await assert.rejects(
    closeAndCheckBrowserEvents({
      close: async () => {
        setImmediate(() => failureLatch.fail('Delayed renderer crash'));
      },
      getFailure: failureLatch.getFailure,
      stopTracing: async () => {},
    }),
    /Delayed renderer crash/
  );
});

test('overlapping samples share one task and await its result', async () => {
  let finishSample;
  let sampleCount = 0;
  const sample = createSingleFlight(
    () =>
      new Promise((resolve) => {
        sampleCount += 1;
        finishSample = resolve;
      })
  );

  const first = sample.run();
  const second = sample.run();

  assert.equal(first, second);
  assert.equal(sample.wait(), first);
  await Promise.resolve();
  assert.equal(sampleCount, 1);

  finishSample('complete');
  assert.deepEqual(await Promise.all([first, second]), [
    'complete',
    'complete',
  ]);

  const third = sample.run();
  assert.notEqual(third, first);
  await Promise.resolve();
  assert.equal(sampleCount, 2);
  finishSample('next');
  assert.equal(await third, 'next');
});

test('fixture output supports exact and partial expectations', () => {
  assert.equal(
    matchesExpectedCompletion(' Paris.', {
      expectedCompletion: ' Paris.',
    }),
    true
  );
  assert.equal(
    matchesExpectedCompletion(' Paris is the capital.', {
      expectedCompletionSubstring: 'Paris',
    }),
    true
  );
  assert.equal(
    matchesExpectedCompletion(' London.', {
      expectedCompletionSubstring: 'Paris',
    }),
    false
  );
});
