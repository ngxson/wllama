/**
 * Test-only helper: hooks into `globalThis.fetch` to simulate network events
 * (dropped connection, stalled download, missing headers, HTTP errors...).
 *
 * Requests that match no rule go to the real network. Requests made from
 * inside a Worker are not affected.
 *
 * The hook is removed automatically when the current test finishes.
 *
 * Example:
 *   const net = hookNetwork();
 *   net.on({ method: 'GET', url: SHARD_2 }).cutAfter(1024);
 */
import { onTestFinished } from 'vitest';

export type RequestMatcher =
  | { url?: string | RegExp; method?: string }
  | ((req: Request) => boolean);

/**
 * `next()` performs the real request.
 */
export type InterceptHandler = (
  req: Request,
  next: () => Promise<Response>
) => Response | Promise<Response>;

export interface RecordedRequest {
  method: string;
  url: string;
}

export interface RuleBuilder {
  /** Pass `bytes` bytes of the body, then fail as if the connection dropped */
  cutAfter(bytes: number): void;
  /**
   * Pass `bytes` bytes of the body, then hang until the request is aborted.
   * The returned promise resolves when the download stalls.
   */
  stallAfter(bytes: number): Promise<void>;
  /** Remove the given response headers, e.g. `content-length` */
  dropHeaders(...names: string[]): void;
  /** Respond with the given HTTP status and an empty body */
  respond(status: number): void;
  /** Reject the request, like a DNS or connection error */
  fail(): void;
  /** Custom handler */
  handle(handler: InterceptHandler): void;
}

export interface NetworkHook {
  /** Every request made while the hook is installed, in order */
  readonly requests: RecordedRequest[];
  /**
   * Add a rule. Rules are checked in the order they were added; the first
   * match wins. `times` limits how many requests the rule applies to.
   */
  on(match: RequestMatcher, opts?: { times?: number }): RuleBuilder;
  /** Remove all rules and clear `requests`, keep the hook installed */
  reset(): void;
  /** Put back the real fetch */
  restore(): void;
}

interface Rule {
  match: (req: Request) => boolean;
  handler: InterceptHandler;
  remaining: number;
}

let installed: NetworkHook | null = null;

export function hookNetwork(): NetworkHook {
  if (installed) {
    throw new Error('hookNetwork() is already installed');
  }
  const realFetch = globalThis.fetch;
  const rules: Rule[] = [];
  const requests: RecordedRequest[] = [];

  globalThis.fetch = async (input, init) => {
    const req = new Request(input, init);
    requests.push({ method: req.method, url: req.url });
    const next = () => realFetch(req);
    const rule = rules.find((r) => r.remaining > 0 && r.match(req));
    if (!rule) return next();
    rule.remaining--;
    return rule.handler(req, next);
  };

  const hook: NetworkHook = {
    requests,
    on(match, opts = {}) {
      const add = (handler: InterceptHandler) => {
        rules.push({
          match: toPredicate(match),
          handler,
          remaining: opts.times ?? Infinity,
        });
      };
      return {
        cutAfter: (bytes) =>
          add(async (req, next) =>
            limitBody(await next(), bytes, () => {
              throw new TypeError(`network error (simulated): ${req.url}`);
            })
          ),
        stallAfter: (bytes) =>
          new Promise<void>((stalled) =>
            add(async (req, next) =>
              limitBody(await next(), bytes, () => {
                stalled();
                return waitForAbort(req.signal);
              })
            )
          ),
        dropHeaders: (...names) =>
          add(async (_req, next) => {
            const res = await next();
            const headers = new Headers(res.headers);
            names.forEach((n) => headers.delete(n));
            return new Response(res.body, {
              status: res.status,
              statusText: res.statusText,
              headers,
            });
          }),
        respond: (status) => add(() => new Response(null, { status })),
        fail: () =>
          add((req) => {
            throw new TypeError(`network error (simulated): ${req.url}`);
          }),
        handle: add,
      };
    },
    reset() {
      rules.length = 0;
      requests.length = 0;
    },
    restore() {
      if (installed !== hook) return;
      globalThis.fetch = realFetch;
      installed = null;
    },
  };

  installed = hook;
  onTestFinished(() => hook.restore());
  return hook;
}

function toPredicate(match: RequestMatcher): (req: Request) => boolean {
  if (typeof match === 'function') return match;
  return (req) => {
    if (match.method && req.method !== match.method.toUpperCase()) {
      return false;
    }
    if (typeof match.url === 'string') return req.url === match.url;
    if (match.url instanceof RegExp) return match.url.test(req.url);
    return true;
  };
}

/**
 * Pass the first `bytes` bytes of the body through, then call `onLimit`.
 * If `onLimit` throws (or its promise rejects), the body stream errors.
 */
function limitBody(
  res: Response,
  bytes: number,
  onLimit: () => Promise<void> | void
): Response {
  if (!res.body) return res;
  const reader = res.body.getReader();
  let sent = 0;
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (sent >= bytes) {
        reader.cancel().catch(() => {});
        await onLimit();
        controller.close();
        return;
      }
      const { done, value } = await reader.read();
      if (done) {
        controller.close();
        return;
      }
      const chunk = value.subarray(0, bytes - sent);
      sent += chunk.byteLength;
      controller.enqueue(chunk);
    },
    cancel(reason) {
      return reader.cancel(reason);
    },
  });
  return new Response(body, {
    status: res.status,
    statusText: res.statusText,
    headers: res.headers,
  });
}

function waitForAbort(signal: AbortSignal): Promise<never> {
  return new Promise((_, reject) => {
    if (signal.aborted) return reject(signal.reason);
    signal.addEventListener('abort', () => reject(signal.reason), {
      once: true,
    });
  });
}
