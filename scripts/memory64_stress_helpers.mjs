import { resolve } from 'node:path';
import { setImmediate as waitForImmediate } from 'node:timers/promises';

const decodeMountInfoPath = (value) =>
  value.replace(/\\([0-7]{3})/g, (_, octal) =>
    String.fromCharCode(Number.parseInt(octal, 8))
  );

/**
 * Resolve the cgroup directory visible through a matching mountinfo entry.
 * `mountFound` distinguishes a host without that controller from a namespaced
 * layout that cannot be mapped for diagnostic reporting.
 */
export const resolveCgroupMount = (
  mountInfo,
  fileSystem,
  cgroupPath,
  requiredOption = null
) => {
  let mountFound = false;

  for (const line of mountInfo.split('\n')) {
    const separator = line.indexOf(' - ');
    if (separator === -1) continue;

    const afterSeparator = line.slice(separator + 3).split(' ');
    const superOptions = afterSeparator.slice(2).join(',').split(',');
    if (
      afterSeparator[0] !== fileSystem ||
      (requiredOption && !superOptions.includes(requiredOption))
    ) {
      continue;
    }
    mountFound = true;

    const beforeSeparator = line.slice(0, separator).split(' ');
    const mountRoot = decodeMountInfoPath(beforeSeparator[3]);
    const mountPoint = decodeMountInfoPath(beforeSeparator[4]);
    const isWithinMount =
      mountRoot === '/' ||
      cgroupPath === mountRoot ||
      cgroupPath.startsWith(`${mountRoot}/`);
    if (!mountPoint || !isWithinMount) continue;

    const relativePath = cgroupPath.slice(mountRoot.length);
    const relativeFromMount = relativePath.startsWith('/')
      ? `.${relativePath}`
      : `./${relativePath}`;
    const path = resolve(mountPoint, relativeFromMount);
    if (path === mountPoint || path.startsWith(`${mountPoint}/`)) {
      return { mountFound, path };
    }
  }

  return { mountFound, path: null };
};

/**
 * Create a single-assignment failure channel for asynchronous run events.
 */
export const createFailureLatch = () => {
  let failure = null;
  let rejectFailure;
  const failureEvent = new Promise((_, reject) => {
    rejectFailure = reject;
  });
  failureEvent.catch(() => {});

  const fail = (message) => {
    if (failure) return;
    failure = message instanceof Error ? message : new Error(message);
    rejectFailure(failure);
  };
  const throwIfFailed = () => {
    if (failure) throw failure;
  };
  const guard = async (promise) => {
    throwIfFailed();
    return await Promise.race([promise, failureEvent]);
  };

  return {
    fail,
    getFailure: () => failure,
    guard,
    throwIfFailed,
  };
};

/**
 * Coalesce overlapping calls so every caller awaits the same active task.
 */
export const createSingleFlight = (task) => {
  let activeTask = null;

  const run = () => {
    if (!activeTask) {
      activeTask = Promise.resolve()
        .then(task)
        .finally(() => {
          activeTask = null;
        });
    }
    return activeTask;
  };

  return {
    run,
    wait: () => activeTask ?? Promise.resolve(),
  };
};

/**
 * Check generated text against the exact or partial output declared by a fixture.
 */
export const matchesExpectedCompletion = (
  completion,
  { expectedCompletion, expectedCompletionSubstring }
) =>
  typeof completion === 'string' &&
  completion.length > 0 &&
  (!expectedCompletion || completion === expectedCompletion) &&
  (!expectedCompletionSubstring ||
    completion.includes(expectedCompletionSubstring));

/**
 * Close the browser event source before a result snapshots fatal events.
 */
export const closeAndCheckBrowserEvents = async ({
  close,
  getFailure,
  onClosed = () => {},
  stopTracing,
}) => {
  await stopTracing().catch(() => {});
  await close();
  onClosed();

  // Playwright can resolve close while a CDP callback is already queued for
  // the next event-loop turn. Let that callback reach the failure latch before
  // the runner snapshots its final result.
  await waitForImmediate();

  const failure = getFailure();
  if (failure) throw failure;
};
