<script lang="ts">
	// replaces the upstream models manager: wllama downloads the models in the browser
	import WllamaModels from '#lib/wllama/components/WllamaModels.svelte';
	import { APP_NAME } from '#lib/constants/index.ts';

	const LINK_CLASS = 'text-(--fg-default) underline underline-offset-2';
</script>

<svelte:head>
	<title>Models - {APP_NAME}</title>
</svelte:head>

<!-- the full width element scrolls, so the scroll bar is on the right edge of the page -->
<div class="h-dvh w-full overflow-y-auto">
	<div
		class="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 pt-[calc(var(--page-top)+1rem)] md:px-8 md:pt-12"
	>
		<div class="flex min-w-0 flex-col gap-1">
			<h1 class="text-heading tracking-tight">Wllama</h1>

			<p class="text-body-sm text-muted-foreground">
				This is a demo of
				<a
					class={LINK_CLASS}
					href="https://github.com/ngxson/wllama"
					rel="noopener noreferrer"
					target="_blank">wllama</a
				>, based on the source code of
				<a
					class={LINK_CLASS}
					href="https://github.com/ggml-org/llama.cpp/tree/master/ui"
					rel="noopener noreferrer"
					target="_blank">llama-ui</a
				>, the web UI of llama.cpp.
				<br />
				The models run in your browser: they are downloaded once, then cached in the browser storage.
			</p>
		</div>

		<WllamaModels />
	</div>
</div>
