// runs before the app starts: the llama-server API is answered in the page by wllama
import { installFetchShim } from '#lib/wllama/fetch-shim.ts';

installFetchShim();

// opening the site shows the Models page; this runs once, before the hash router reads the URL,
// so "New chat" (#/) still opens the chat, and a link to a chat or a page is kept
if (window.location.hash === '' || window.location.hash === '#/') {
	history.replaceState(history.state, '', '#/models');
}
