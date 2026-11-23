<script lang="ts">
	import { Wllama } from '@wllama/wllama';
	import { Check, X } from '@lucide/svelte';
	import { WLLAMA_VERSION } from 'virtual:wllama-build';

	// same checks as wllama (src/utils.ts), they are not exported by the library
	const isSupportJSPI = () => !!(WebAssembly as unknown as { Suspending?: unknown }).Suspending;
	const isSupportWebGPU = () => !!(navigator as unknown as { gpu?: unknown }).gpu;
	// a SharedArrayBuffer that can be sent to a worker needs cross-origin isolation (COOP/COEP headers)
	const isSupportMultiThread = () => {
		try {
			new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
			return true;
		} catch {
			return false;
		}
	};

	const rows: { label: string; value: string | boolean }[] = [
		{ label: 'wllama', value: WLLAMA_VERSION },
		{ label: 'llama.cpp', value: Wllama.getLibllamaVersion() },
		{ label: 'WebGPU', value: isSupportWebGPU() },
		{ label: 'JSPI', value: isSupportJSPI() },
		{ label: 'Multithread', value: isSupportMultiThread() }
	];
</script>

<section class="flex flex-col gap-3">
	<h2 class="text-title">Runtime info</h2>

	<div class="flex flex-wrap gap-x-6 gap-y-2 text-body-sm">
		{#each rows as row (row.label)}
			<span class="flex items-center gap-1.5">
				<span class="text-muted-foreground">{row.label}</span>
				{#if typeof row.value === 'string'}
					<span class="font-mono">{row.value}</span>
				{:else if row.value}
					<Check aria-label="supported" class="size-icon-sm text-green-600 dark:text-green-400" />
				{:else}
					<X aria-label="not supported" class="size-icon-sm text-red-600 dark:text-red-400" />
				{/if}
			</span>
		{/each}
	</div>
</section>
