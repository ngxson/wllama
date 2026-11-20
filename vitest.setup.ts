declare const __HF_TOKEN__: string;
declare const __COMPAT__: boolean;

// compat test run: hide JSPI on the page, so needCompat() is true (the worker keeps it, asyncify does not use it)
if (__COMPAT__) {
  delete (WebAssembly as any).Suspending;
  delete (WebAssembly as any).promising;
}

// with a token, HF does not rate limit the downloads (the CI uses a read-only one)
if (__HF_TOKEN__) {
  const fetchOrig = globalThis.fetch;
  globalThis.fetch = (input, init) => {
    const url = new URL(
      input instanceof Request ? input.url : String(input),
      location.href
    );
    if (url.hostname !== 'huggingface.co') {
      return fetchOrig(input, init);
    }
    const headers = new Headers(
      init?.headers ?? (input instanceof Request ? input.headers : undefined)
    );
    if (!headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${__HF_TOKEN__}`);
    }
    return fetchOrig(input, { ...init, headers });
  };
}
