declare const __HF_TOKEN__: string;

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
