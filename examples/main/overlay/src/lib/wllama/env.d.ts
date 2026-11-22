declare module 'virtual:wllama-build' {
	export const WLLAMA_VERSION: string;
	export const WLLAMA_WASM: string;
	export const WLLAMA_COMPAT: { wasm: string } | 'default';
}
