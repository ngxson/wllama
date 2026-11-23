// sends the same-origin llama-server API requests to routes.ts, the others go to the network

import { handleApiRequest, isApiPath } from './routes';

export function installFetchShim() {
	const originalFetch = window.fetch;
	// the app uses relative paths and a hash router, so the API lives under the page folder
	const appBase = new URL('./', window.location.href).pathname;

	window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
		const url = new URL(input instanceof Request ? input.url : String(input), window.location.href);
		if (url.origin === window.location.origin && url.pathname.startsWith(appBase)) {
			const path = '/' + url.pathname.slice(appBase.length);
			if (isApiPath(path)) {
				return handleApiRequest(new Request(input, init), path);
			}
		}
		return originalFetch(input, init);
	};
}
