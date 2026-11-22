// wraps the upstream vite.config.ts, so it does not need a patch
import upstreamConfig from './vite.config';
import { existsSync, readFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import { defineConfig, type Plugin, type UserConfig } from 'vite';

const __dirname = dirname(fileURLToPath(import.meta.url));
// the app is staged in examples/main/.build/ui/apps/chat
const WLLAMA_ROOT = process.env.WLLAMA_ROOT ?? resolve(__dirname, '../../../../../..');
const WLLAMA_WASM = resolve(WLLAMA_ROOT, 'esm/wasm/wllama.wasm');
const COMPAT_WASM = resolve(WLLAMA_ROOT, 'compat/wasm/wllama.wasm');

// virtual:wllama-build gives the wasm URLs and the version
function wllamaBuildPlugin(): Plugin {
	const VIRTUAL_ID = 'virtual:wllama-build';
	return {
		load(id) {
			if (id !== '\0' + VIRTUAL_ID) return;
			const version = JSON.parse(
				readFileSync(resolve(WLLAMA_ROOT, 'package.json'), 'utf8')
			).version;
			const hasCompat = existsSync(COMPAT_WASM);
			if (!hasCompat) {
				console.warn(
					'[wllama] compat wasm not found, the compat build will be loaded from the CDN'
				);
			}
			return [
				`import wasm from '${WLLAMA_WASM}?url';`,
				hasCompat ? `import compatWasm from '${COMPAT_WASM}?url';` : 'const compatWasm = null;',
				`export const WLLAMA_VERSION = ${JSON.stringify(version)};`,
				`export const WLLAMA_WASM = wasm;`,
				`export const WLLAMA_COMPAT = compatWasm ? { wasm: compatWasm } : 'default';`
			].join('\n');
		},
		name: 'wllama-build',
		resolveId(id) {
			if (id === VIRTUAL_ID) return '\0' + VIRTUAL_ID;
		}
	};
}

export default defineConfig(async (env) => {
	const config = (await (typeof upstreamConfig === 'function'
		? upstreamConfig(env)
		: upstreamConfig)) as UserConfig;

	// no llama-server behind the app: the API is answered in the page (src/lib/wllama)
	delete config.server?.proxy;
	delete (config as { test?: unknown }).test;

	config.plugins = [...(config.plugins ?? []), wllamaBuildPlugin()];
	config.server!.fs!.allow!.push(WLLAMA_ROOT);
	// the wllama worker code is inlined as a string, do not let vite pre-bundle it again
	config.optimizeDeps = { ...config.optimizeDeps, exclude: ['@wllama/wllama'] };

	return config;
});
